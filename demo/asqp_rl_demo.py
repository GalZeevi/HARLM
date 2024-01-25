from itertools import chain, cycle
import ipywidgets as widgets
import json
import pandas as pd
from IPython.display import clear_output
from IPython.display import display
from IPython.display import display_html
import sqlparse

with open('assets/queries.sql', 'r') as queries_file:
    queries = [q.strip() for q in queries_file.readlines()]


class AsqpInstance:
    def __init__(self):
        self.score = 0
        self.index = -1

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
                                              correct_answer))

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

    def demonstrate_asqp_rl(self):
        df1, df2 = self.get_dfs()
        AsqpInstance.display_side_by_side(df1, df2)
        self.choose_answers_button()
        self.reveal_answers_button()

    @staticmethod
    def display_side_by_side(*args, titles=cycle([''])):
        html_str = ''
        for df, title in zip(args, chain(titles, cycle(['</br>']))):
            html_str += '<th style="text-align:center"><td style="vertical-align:top">'
            html_str += f'<h2 style="text-align: center;">{title}</h2>'
            html_str += df.sample(frac=1).to_html(max_rows=20, index=False).replace('table', 'table style="display:inline"')
            html_str += '</td></th>'
        display_html(html_str, raw=True)

    def create_multiplechoice_widget(self, description, options, answer):
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
            with output:
                clear_output()
                print('Answer saved.')
            return

        check = widgets.Button(description="submit")
        check.on_click(check_selection)

        return widgets.VBox([description_out, alternativ, check, output])

    def reveal_results(self):
        print(f'You were correct in {self.score} out of {len(queries)} questions')


def demonstrate_asqp_rl(asqprl: AsqpInstance):
    df1, df2 = asqprl.get_dfs()
    AsqpInstance.display_side_by_side(df1, df2)
    asqprl.choose_answers_button()
    asqprl.reveal_answers_button()
