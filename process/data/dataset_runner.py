from processor_factory import get_handler

def run(dataset_name):
    handler = get_handler(dataset_name)
    result = handler.load()
    print("处理结果:", result)

if __name__ == "__main__":
    run("darpa")