import csv


def read_data(file_name):
    data = []
    with open(file_name, 'r') as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            data.append({"question_text": row[0], "choices": row[1:-1], "correct_answer": row[-1]})
    return data
