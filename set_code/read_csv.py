import csv


def read_csv_file(file_path):
    try:
        with open(file_path, 'r') as csvfile:
            # 创建CSV读取对象
            csv_reader = csv.reader(csvfile)

            # 读取并显示数据
            for row in csv_reader:
                print(row)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


# 指定要读取的CSV文件路径
csv_file_path = '../data_label.csv'

# 调用函数读取并显示CSV文件内容
read_csv_file(csv_file_path)
