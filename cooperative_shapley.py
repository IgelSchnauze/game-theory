import PySimpleGUI as sg
import numpy as np
from math import factorial as fact
from itertools import combinations as combo

numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def vec_Shapley(N, K_coalite, V_k_func):
    w = np.zeros(N)
    for player_ind in range(N):
        for i, coalite in enumerate(K_coalite):
            if player_ind+1 in coalite:
                k = len(coalite)
                w_i = (fact(N-k) * fact(k-1)) / fact(N)
                v_without = list(filter(lambda x: x != player_ind+1, K_coalite[i]))
                w_i *= (V_k_func[i] - V_k_func[K_coalite.index(v_without)])
                w[player_ind] += w_i
    return w


def check_superadd(K_coalite, V_k_func):
    is_okay = True
    for i in range(len(K_coalite)-1):
        for j in range(len(K_coalite)):
            new_coalite = list(set(K_coalite[i] + K_coalite[j]))
            if i == j or len(K_coalite[i] + K_coalite[j]) > len(new_coalite):
                continue
            v_i_j = V_k_func[i] + V_k_func[j]
            v_ij = V_k_func[K_coalite.index(new_coalite)]

            if v_i_j > v_ij:
                is_okay = False
                break
        if not is_okay:
            break
    return is_okay


if __name__ == '__main__':
    N = 0
    sg.theme('DarkBrown')
    font_window = ('Arial 12 bold')
    font_input = ('Arial 12')
    layout_start = [
        [sg.Text('Input num of players:', font=font_window),
         sg.InputText(size=(5,1), font=font_input,
                      key='-num-')],
        [sg.Button('Ok', font=font_window)]
    ]
    window_start = sg.Window('Shapley vector from KVA', layout_start)

    while True:
        event, values = window_start.read()
        if event in (None, 'Exit'):
            break
        if event == "Ok":
            N = int(values["-num-"])
            break

    K = sum([list(map(list, combo([j+1 for j in range(N)], i))) for i in range(N + 1)], [])

    layout = [
        [sg.Input(f'{coalite}', size=(8,1), font=font_window, pad=(0,0),
                  readonly=True, disabled_readonly_background_color='gray32') for coalite in K],
        [sg.Input(size=(8,1), font=font_input, pad=(0,0), key=f'{i}') for i in range(len(K))],
        [sg.Button('Share-out', font=font_window)],
        [sg.Text('\n')],
        [[sg.Text(f'Sum for {i} player = ', font=font_window, key=f'w{i}')] for i in range(N)]
    ]
    window = sg.Window('Shapley vector from KVA', layout)

    while True:
        event, values = window.read()
        if event in (None, 'Exit') or N == 0:
            break

        if event == 'Share-out':
            V_k = np.zeros(len(K))
            is_input_okey = True

            for i in range(len(K)):
                v_i = values[f'{i}'].replace(' ', '')
                if not all([v in numbers for v in v_i]):
                    sg.PopupOK('V(k) value can be only number!')
                    is_input_okey = False
                    break
                V_k[i] = int(v_i)
            if not is_input_okey:
                continue

            is_input_okey = check_superadd(K, V_k)
            if not is_input_okey:
                sg.PopupOK('Condition is breaking')
                continue

            vec = vec_Shapley(N, K, V_k)

            for i in range(N):
                window[f'w{i}'].update(f'Sum for {i} player = {vec[i]}')
