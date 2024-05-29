from itertools import chain, cycle
import ipywidgets as widgets
import json
import pandas as pd
from IPython.display import clear_output
from IPython.display import display
from IPython.display import display_html
import sqlparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

with open('assets/queries.sql', 'r') as queries_file:
    queries = [q.strip() for q in queries_file.readlines()]


class AsqpInstance:
    def __init__(self, index=-1, name='sigmod_demo', num_queries=5):
        self.score = 0
        self.answers = [0] * len(queries)
        self.index = index
        self.name = name
        self.num_queries = num_queries

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
        button = widgets.Button(description="Show correct answer")
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
                print(f'Answer saved.')
            return

        check = widgets.Button(description="submit")
        check.on_click(check_selection)

        return widgets.VBox([description_out, alternativ, check, output])

    def reveal_results(self):
        print(f'You were correct in {sum(self.answers)} out of {self.num_queries} questions')

    @staticmethod
    def save_answers(user, name, answers, score, num_queries):
        my_file = Path(f"assets/scores/{name}.csv")
        if my_file.is_file():
            lines_to_write = []
        else:
            lines_to_write = [
                ','.join([*[f'q{query_num}' for query_num in [i + 1 for i in range(num_queries)]], 'total', 'user'])]

        lines_to_write.append(f'{",".join([str(ans) for ans in answers])},{score},{user}')

        with open(f"assets/scores/{name}.csv", 'a') as the_file:
            for line in lines_to_write:
                the_file.write(f'{line}\n')

    def save_answers_button(self):
        input = widgets.Text(value="state your name")

        button = widgets.Button(description="Save answers", disabled=True)
        output = widgets.Output()

        display(input, button, output)
        name = self.name
        answers = self.answers
        score = self.score
        num_queries = self.num_queries

        def on_button_clicked(b):
            with output:
                if not input.value:
                    print('Please state your name first')
                else:
                    clear_output()
                    AsqpInstance.save_answers(input.value, name, answers, sum(answers), num_queries)
                    print('Done')

        button.on_click(on_button_clicked)

        def value_changed(change):
            button.disabled = not bool(change.new)

        input.observe(value_changed, "value")


def demonstrate_asqp_rl(asqprl: AsqpInstance):
    df1, df2 = asqprl.get_dfs()
    AsqpInstance.display_side_by_side(df1, df2)
    asqprl.choose_answers_button()
    asqprl.reveal_answers_button()
    # asqprl.save_answers_button()


def show_scoreboard(asqprl: AsqpInstance):
    button = widgets.Button(description="Show scoreboard")
    output = widgets.Output()
    display(button, output)

    def _show_scoreboard(b):
        clear_output()
        with open(f"assets/scores/{asqprl.name}.csv", 'r') as file:
            # Read all lines and strip newline characters
            data_list = [line.strip() for line in file]

        data_list = data_list[1:]  # remove header
        # take only the user and total from each row
        data_list = [[s.strip() for s in row.split(',')][-2:] for row in data_list]
        df = pd.DataFrame(data_list, columns=['total', 'user'])
        df['total'] = df['total'].astype(int)

        # Set style and color palette
        sns.set(style='whitegrid')
        palette = sns.color_palette("crest")

        # Create the bar chart
        plt.figure(figsize=(8, 4))
        df = df.groupby(['user']).sum().reset_index()
        barplot = sns.barplot(x='user', y='total', data=df, palette=palette)

        # Add titles and labels
        plt.title('Scoreboard', fontsize=16)
        plt.xlabel('User', fontsize=14)
        plt.ylabel('Score', fontsize=14)

        # Customize the appearance
        barplot.set_yticks(range(0, 5, 1))
        sns.despine(left=True, bottom=True)

        # Show the plot
        plt.show()
        display(b)

    button.on_click(_show_scoreboard)


def reveal_results(asqprl: AsqpInstance):
    button = widgets.Button(description="Reveal results")
    output = widgets.Output()
    display(button, output)

    def onclick(b):
        clear_output()
        asqprl.reveal_results()

    button.on_click(onclick)
