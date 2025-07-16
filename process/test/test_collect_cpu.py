import json
import subprocess
import threading
import time
import psutil
import sys
import queue
import os

# 创建一个线程安全的队列
print_queue = queue.Queue()

def printer_worker():
    while True:
        message = print_queue.get()
        if message is None:
            break
        print(message)
        print_queue.task_done()  #通知队列，刚刚取出的那个任务已经被处理完毕。

class CPUMonitor:
    def __init__(self, interval=1, output_file="cpu_usage_log.json"):
        self.sample_interval = interval
        self.output_file = output_file
        self.cpu_usage = []
        self.monitoring = False

    def collect_cpu_usage(self):
        psutil.cpu_percent(interval=None) # 第一次调用它会返回0.0并建立一个时间基准。
        while self.monitoring:
            loop_start_time = time.monotonic()  #使用monotonic时钟记录循环开始的精确时间点。
            usage = psutil.cpu_percent(interval=None) #非阻塞地获取自上次调用以来的CPU平均使用率。
            timestamp = time.time()
            self.cpu_usage.append({"timestamp": timestamp, "cpu_percent": usage})
            print_queue.put(f"[CPU MONITOR]: {usage}%")  #将格式化好的字符串放入共享的 print_queue 中
            work_duration = time.monotonic() - loop_start_time
            sleep_time = self.sample_interval - work_duration  #精确计算出本次循环需要休眠多久，才能使得整个循环的周期恰好是我们设定的sample_interval
            if sleep_time > 0:
                time.sleep(sleep_time)

    def save_to_file(self):
        with open(self.output_file, "w") as f:
            json.dump(self.cpu_usage, f, indent=4)
        print(f"\n[INFO] CPU usage data saved to {self.output_file}")

    def start_monitoring(self):
        self.monitoring = True  #创建一个新的守护线程来运行 collect_cpu_usage 方法。daemon=True意味着如果主程序退出了，这个线程也会被强制终止，不会阻止程序关闭
        monitor_thread = threading.Thread(target=self.collect_cpu_usage, daemon=True)
        monitor_thread.start()

    def stop_monitoring(self):
        self.monitoring = False

def stream_output_to_queue(stream, prefix):
    """将子进程的输出流逐行放入打印队列"""
    try:
        for line in iter(stream.readline, ''):  #持续调用 stream.readline()，直到它返回一个空字符串（代表流已结束）
            print_queue.put(f"{prefix} {line.strip()}")  #将从子进程读到的每一行，加上一个前缀（如 [test.py]）后，放入共享的 print_queue
    finally:
        # 确保流在线程结束时关闭
        stream.close()

def run_xxx_with_cpu_monitor():
    printer_thread = threading.Thread(target=printer_worker, daemon=True)
    printer_thread.start()  #启动“打印机”线程

    cpu_monitor = CPUMonitor(interval=1, output_file="cpu_usage_log.json")
    cpu_monitor.start_monitoring()  #创建并启动CPU监控器

    process = None
    try:
        command = [sys.executable, "process/test.py"]
        working_dir = ".."

        env = os.environ.copy()
        #强制子进程的Python I/O使用UTF-8编码
        env['PYTHONIOENCODING'] = 'utf-8'
        # ==================================================

        print_queue.put(f"[INFO] Starting subprocess: {' '.join(command)}")

        process = subprocess.Popen(  #subprocess.Popen: 启动子进程
            command, cwd=working_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE,  #将子进程的正常输出和错误输出重定向到内存中的“管道”，而不是直接显示在屏幕上
            text=True, encoding='utf-8', # 父进程继续使用UTF-8解码
            bufsize=1,  # 开启行缓冲模式。这意味着子进程每输出一个换行符，数据就会被立即发送到管道，而不是等缓冲区满了再说。这对实时性至关重要。
            env=env # <--- 将修改后的环境变量传递给子进程
        )
        #创建并启动两个“搬运工”线程，分别负责从子进程的stdout和stderr管道中读取数据并放入队列。
        stdout_thread = threading.Thread(target=stream_output_to_queue, args=(process.stdout, "[test.py]"), daemon=True)
        stderr_thread = threading.Thread(target=stream_output_to_queue, args=(process.stderr, "[test.py-ERROR]"), daemon=True)
        stdout_thread.start()
        stderr_thread.start()

        print_queue.put("[INFO] Monitoring CPU and subprocess output...")
       #主线程执行到这里，会暂停并等待，直到 test.py 子进程完全执行结束。在此期间，所有后台线程（打印机、CPU监控、输出读取）都在并行地忙碌着
        process.wait()
        #子进程结束后，可能管道里还有一些最后的输出没被读完。这两行代码会等待“搬运工”线程完成它们的工作，确保所有输出都被处理
        stdout_thread.join()
        stderr_thread.join()

        print_queue.put(f"\n[INFO] Subprocess 'test.py' finished with exit code {process.returncode}.")

    except Exception as e:
        print_queue.put(f"[ERROR] An unexpected error occurred: {e}")
    finally:
        if process and process.poll() is None:
            process.terminate()

        cpu_monitor.stop_monitoring()
        time.sleep(1)

        print_queue.join()  #阻塞，直到队列中所有已放入的项目都被task_done()标记为完成。这确保了所有消息都被打印了出来。
        print_queue.put(None)  #向队列放入“下班”信号
        printer_thread.join(timeout=2)  #等待打印机线程接收到None并正常退出。

        cpu_monitor.save_to_file()

if __name__ == "__main__":
    run_xxx_with_cpu_monitor()