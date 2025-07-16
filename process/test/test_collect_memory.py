# 导入所有需要的库
import json             # 用于处理JSON数据的序列化（写入文件）
import subprocess       # 用于创建和管理子进程（运行test.py）
import threading        # 用于创建和管理并发执行的线程
import time             # 用于时间相关操作，如延时(sleep)和精确计时(monotonic)
import psutil           # 用于获取系统和进程的资源使用情况（如内存）
import sys              # 用于与Python解释器交互（获取解释器路径）
import queue            # 用于创建线程安全的队列，是多线程通信的核心
import os               # 用于与操作系统交互（操作环境变量）

# --- 1. 核心基础设施：打印队列与专职打印机线程 ---
# 这个部分是通用设计，用于解决所有多线程打印冲突问题。

# 创建一个全局的、线程安全的队列。它将作为所有需要打印的信息的中央“消息中转站”。
print_queue = queue.Queue()

def printer_worker():
    """
    这个函数是“专职打印机”线程的工作内容。
    它的唯一职责就是不断地从print_queue中取出消息并打印到控制台。
    """
    while True:
        # 从队列中获取一条消息。如果队列是空的，这个调用会阻塞（暂停），直到有新消息进入。
        message = print_queue.get()

        # 我们约定，None是一个特殊的“停止信号”。当收到它时，就意味着工作结束了。
        if message is None:
            break

        # 将取出的消息打印到屏幕上。这是整个程序中唯一直接调用print()的地方。
        print(message)

        # 通知队列，刚才取出的这个任务已经被处理完毕。这对于后面的print_queue.join()至关重要。
        print_queue.task_done()

# --- 2. 内存监控模块（只监控特定进程） ---
class MemoryMonitor:
    """
    一个专门用于监控特定子进程内存使用情况的类。
    它被设计成一个“生产者”，不断地生成内存数据。
    """
    def __init__(self, interval=1, output_file="memory_usage_log.json"):
        # self.sample_interval: 我们期望的采样周期（例如，每1秒一次）。
        self.sample_interval = interval
        # self.output_file: 保存最终监控数据的JSON文件名。
        self.output_file = output_file
        # self.memory_usage: 一个列表，用于在内存中暂存所有收集到的数据点。
        self.memory_usage = []
        # self.monitoring: 一个布尔标志，用于控制监控循环的启动和停止。
        self.monitoring = False
        # self.target_pid: 用于存储目标子进程的进程ID (PID)。
        self.target_pid = None
        # self.process_handle: 用于存储psutil.Process对象，避免在循环中重复创建，提高效率。
        self.process_handle = None

    def set_target_pid(self, pid):
        """主线程调用此方法来设置目标PID，并正式“激活”监控。"""
        self.target_pid = pid
        try:
            # 根据PID创建一个psutil的Process对象，后续所有操作都通过这个句柄进行。
            self.process_handle = psutil.Process(pid)
        except psutil.NoSuchProcess:
            # 这是一个健壮性检查：如果在设置目标时，该进程就已经不存在了。
            print_queue.put(f"[ERROR] Process with PID {pid} does not exist when setting target.")
            self.monitoring = False # 停止监控

    def collect_memory_usage(self):
        """在独立线程中运行的核心监控方法。"""
        # 阶段1: 等待PID被设置。此线程启动后，会先在这里“空转”等待。
        # self.monitoring在start_monitoring时已设为True，所以这个循环会运行。
        while self.process_handle is None:
            if not self.monitoring: # 如果在等待期间被主线程命令停止，则优雅退出。
                return
            time.sleep(0.1) # 短暂休眠，避免空转消耗过多CPU。

        # 阶段2: PID已设置，开始真正的监控循环。
        while self.monitoring:
            try:
                # 记录循环开始的精确时间点，用于后续的精确延时计算。
                loop_start_time = time.monotonic()

                # --- 核心逻辑：获取特定进程的内存 ---
                # memory_info().rss (Resident Set Size) 是进程实际占用的物理内存，是通常最关心的指标。
                memory_used = self.process_handle.memory_info().rss / (1024 * 1024)

                # 获取当前的标准时间戳，用于记录。
                timestamp = time.time()
                # 将数据点作为一个字典，追加到数据列表中。
                self.memory_usage.append({"timestamp": timestamp, "memory_used_MB": memory_used})

                # 将格式化好的消息字符串放入共享的打印队列。
                print_queue.put(f"[MEMORY MONITOR (PID:{self.target_pid})]: {memory_used:.2f} MB")

                # --- 精确控制频率 ---
                work_duration = time.monotonic() - loop_start_time
                sleep_time = self.sample_interval - work_duration
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except psutil.NoSuchProcess:
                # 健壮性处理：如果在监控过程中，目标进程突然结束了，会触发此异常。
                print_queue.put(f"[INFO] Target process {self.target_pid} has ended. Stopping memory monitor.")
                break # 正常退出监控循环。
            except Exception as e:
                # 捕获其他未知错误，防止监控线程崩溃。
                print_queue.put(f"[ERROR] Memory monitor encountered an error: {e}")
                break

        # 无论循环如何退出（正常停止或因错误中断），都将监控标志设为False，确保状态一致。
        self.monitoring = False

    def save_to_file(self):
        """将收集到的所有内存数据以JSON格式保存到文件。"""
        with open(self.output_file, "w") as f:
            json.dump(self.memory_usage, f, indent=4)
        print(f"\n[INFO] Memory usage data saved to {self.output_file}")

    def start_monitoring(self):
        """启动内存监控线程，并设置监控标志为True。"""
        # --- 关键修复：必须在这里设置监控标志！ ---
        # 这确保了线程启动后，会进入“等待PID”的循环，而不是立即退出。
        self.monitoring = True
        # 创建并启动一个守护线程来执行监控任务。
        monitor_thread = threading.Thread(target=self.collect_memory_usage, daemon=True)
        monitor_thread.start()

    def stop_monitoring(self):
        """从外部停止内存监控。"""
        self.monitoring = False

# --- 3. 子进程输出处理模块 ---
def stream_output_to_queue(stream, prefix):
    """一个通用的函数，负责从一个流中逐行读取数据并放入打印队列。"""
    try:
        # iter(stream.readline, '')是高效的逐行读取方式。
        for line in iter(stream.readline, ''):
            print_queue.put(f"{prefix} {line.strip()}")
    finally:
        # 确保流最终被关闭以释放资源。
        stream.close()

# --- 4. 主执行函数 ---
def run_xxx_with_memory_monitor():
    """程序的总指挥，负责协调所有线程和子进程。"""
    # 步骤1: 启动“专职打印机”线程。
    printer_thread = threading.Thread(target=printer_worker, daemon=True)
    printer_thread.start()

    # 步骤2: 创建内存监控器实例。
    memory_monitor = MemoryMonitor(interval=1, output_file="memory_usage_log.json")
    # 启动监控线程。此时，它会处于“等待PID”的待命状态。
    memory_monitor.start_monitoring()

    process = None
    try:
        # 定义要执行的命令。
        command = [sys.executable, "process/test.py"]
        working_dir = ".."

        # 设置环境变量以解决Windows下的编码问题。
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'

        print_queue.put(f"[INFO] Starting subprocess: {' '.join(command)}")

        # 步骤3: 使用Popen非阻塞地启动子进程。
        process = subprocess.Popen(
            command, cwd=working_dir,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, # 重定向输出到管道
            text=True, encoding='utf-8', # 以UTF-8文本模式处理管道数据
            bufsize=1, # 开启行缓冲，保证实时性
            env=env    # 应用修改后的环境变量
        )

        # 步骤4: 获取PID并“激活”监控器。
        pid = process.pid # 获取新创建子进程的PID。
        print_queue.put(f"[INFO] Subprocess started with PID: {pid}. Attaching memory monitor.")
        memory_monitor.set_target_pid(pid) # 将PID传递给监控器，使其开始真正的数据采集。

        # 步骤5: 启动线程来处理子进程的输出。
        stdout_thread = threading.Thread(target=stream_output_to_queue, args=(process.stdout, "[test.py]"), daemon=True)
        stderr_thread = threading.Thread(target=stream_output_to_queue, args=(process.stderr, "[test.py-ERROR]"), daemon=True)
        stdout_thread.start()
        stderr_thread.start()

        print_queue.put("[INFO] Monitoring memory of subprocess and its output...")

        # 步骤6: 主线程在此等待，直到子进程执行完毕。
        process.wait()

        # 步骤7: 等待输出处理线程也结束，确保所有输出都被捕获。
        stdout_thread.join()
        stderr_thread.join()

        print_queue.put(f"\n[INFO] Subprocess 'test.py' (PID:{pid}) finished with exit code {process.returncode}.")

    except Exception as e:
        # 捕获任何意外，并打印错误信息。
        print_queue.put(f"[ERROR] An unexpected error occurred: {e}")
    finally:
        # --- 步骤8: 优雅地关闭所有部分，这个块的代码无论如何都会被执行 ---

        # 安全检查：如果子进程还在运行，则终止它。
        if process and process.poll() is None:
            process.terminate()

        # 命令内存监控器停止工作。
        memory_monitor.stop_monitoring()
        time.sleep(1) # 短暂等待，以确保最后一次监控数据能被成功收集。

        # 等待打印队列中的所有消息都被打印完毕。
        print_queue.join()
        # 发送“停止信号”给打印机线程。
        print_queue.put(None)
        # 等待打印机线程完全退出。
        printer_thread.join(timeout=2)

        # 在一切都结束后，安全地将数据保存到文件。
        memory_monitor.save_to_file()

# --- 程序入口 ---
if __name__ == "__main__":
    # 当脚本被直接执行时，调用主函数。
    run_xxx_with_memory_monitor()