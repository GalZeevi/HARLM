from itertools import chain, cycle
import ipywidgets as widgets
import json
import pandas as pd
from IPython.display import clear_output
from IPython.display import display
from IPython.display import display_html
import sqlparse
from pathlib import Path

with open('assets/queries.sql', 'r') as queries_file:
    queries = [q.strip() for q in queries_file.readlines()]


class AsqpInstance:
    def __init__(self, index=-1, name='demo'):
        self.score = 0
        self.answers = [0] * len(queries)
        self.index = index
        self.name = name

    def get_sql(self):
        query = sqlparse.format(queries[self.index], reindent=True, keyword_case='upper')
        return query

    def get_dfs(self):
        return pd.read_csv(f'assets/results/{self.index}/left.csv'), pd.read_csv(
            f'assets/results/{self.index}/right.csv')

    def choose_answers_button(self):
        with open(f'assets/results/{self.index}/answers.json', 'r') as ans_file:
            left_dict, right_dict = json.load(ans_file)
        correct_answer = 0 if left_dict['source'] == 'ASQP-RL' else 1  # 0 if first option, 1 is the second
        display(
            self.create_multiplechoice_widget('Which is true:',
                                              ['Left: ASQP-RL, Right: DB', 'Left: DB, Right: ASQP-RL'],
                                              correct_answer,
                                              self.index))

    @staticmethod
    def get_answers(index):
        with open(f'assets/results/{index}/answers.json', 'r') as ans_file:
            left_dict, right_dict = json.load(ans_file)
        left_ans = 'Left answer from {0} (computed in {1} {2}) and contains a total of {3} rows'.format(
            left_dict['source'],
            left_dict['time'],
            left_dict.get('timeUnits', 'seconds'),
            left_dict['rowCount'])
        right_ans = 'Right answer from {0} (computed in {1} {2}) and contains a total of {3} rows'.format(
            right_dict['source'],
            right_dict['time'],
            right_dict.get('timeUnits', 'seconds'),
            right_dict['rowCount'])
        return f'{left_ans}\n{right_ans}'

    def reveal_answers_button(self):
        button = widgets.Button(description="Reveal answers")
        output = widgets.Output()

        display(button, output)
        index = self.index

        def on_button_clicked(b):
            with output:
                clear_output()
                print(AsqpInstance.get_answers(index))

        button.on_click(on_button_clicked)

    def query_asqp(self, sql):
        self.index += 1

    # def demonstrate_asqp_rl(self):
    #     df1, df2 = self.get_dfs()
    #     AsqpInstance.display_side_by_side(df1, df2)
    #     self.choose_answers_button()
    #     self.save_answers_button()

    @staticmethod
    def display_side_by_side(*args, titles=cycle([''])):
        html_str = ''
        for df, title in zip(args, chain(titles, cycle(['</br>']))):
            html_str += '<th style="text-align:center"><td style="vertical-align:top">'
            html_str += f'<h2 style="text-align: center;">{title}</h2>'
            html_str += df.sample(frac=1).to_html(max_rows=20, index=False).replace('table',
                                                                                    'table style="display:inline"')
            html_str += '</td></th>'
        display_html(html_str, raw=True)

    def create_multiplechoice_widget(self, description, options, answer, index):
        radio_options = [(words, i) for i, words in enumerate(options)]
        alternativ = widgets.RadioButtons(
            options=radio_options,
            description='',
            disabled=False
        )

        description_out = widgets.Output()
        with description_out:
            print(description)

        output = widgets.Output()

        def check_selection(b):
            a = int(alternativ.value)
            if a == answer:
                self.score += 1
                self.answers[index] = 1
            else:
                self.answers[index] = 0
            with output:
                clear_output()
                print('Answer saved.')
            return

        check = widgets.Button(description="submit")
        check.on_click(check_selection)

        return widgets.VBox([description_out, alternativ, check, output])

    def reveal_results(self):
        print(f'You were correct in {sum(self.answers)} out of {len(queries)} questions')

    @staticmethod
    def save_answers(name, answers, score):
        my_file = Path(f"assets/scores/{name}.csv")
        if my_file.is_file():
            lines_to_write = []
        else:
            lines_to_write = [
                ','.join([*[f'q{query_num}' for query_num in [i + 1 for i in range(len(queries))]], 'total'])]

        lines_to_write.append(f'{",".join([str(ans) for ans in answers])},{score}')

        with open(f"assets/scores/{name}.csv", 'a') as the_file:
            for line in lines_to_write:
                the_file.write(f'{line}\n')

    def save_answers_button(self):
        button = widgets.Button(description="Finish")
        output = widgets.Output()

        display(button, output)
        name = self.name
        answers = self.answers
        score = self.score

        def on_button_clicked(b):
            with output:
                clear_output()
                AsqpInstance.save_answers(name, answers, sum(answers))
                print('Done')

        button.on_click(on_button_clicked)


def demonstrate_asqp_rl(asqprl: AsqpInstance):
    df1, df2 = asqprl.get_dfs()
    AsqpInstance.display_side_by_side(df1, df2)
    asqprl.choose_answers_button()
    asqprl.save_answers_button()
