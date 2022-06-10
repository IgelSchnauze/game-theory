import numpy as np
import PySimpleGUI as sg

numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def unpack_data(file):
    with open(file, "r") as f:
        m = int(f.readline())
        a = []
        for i in range(m):
            a.append([float(num) for num in f.readline().split()])
        a = np.array(a)
    return a


def simplex_add_process(A):
    m, n = A.shape
    # все ограничения вида неравенств <=1
    b = np.ones(m)
    # начальный единич базис с пом метода доп переменных
    A = np.concatenate((A, np.eye(m)), axis=1)
    A = np.concatenate(([np.array(b)], A.T), axis=0).T  # добавили столбец A[0] (BP_0)
    n_expan = A.shape[1]
    basis_index = np.array([i for i in range(n + 1, n_expan)])  # нач базис из доп переменных

    # все коэф-ты целевой функции = 1 для имеющихся, 0 для доп переменных
    C = np.concatenate((np.ones(n), np.zeros(n_expan - (n+1))))
    C = np.concatenate(([0], C))

    step, final_A, final_basis_index, final_delta = simplex_add_iteration(1, A, basis_index, C, n_expan, m)

    V = 1 / (np.sum(final_A[:, 0]))  # цена игры

    x = final_delta[n+1:] * V
    y = np.zeros(n)
    for i, ind in enumerate(final_basis_index):
        if ind <= n:  # не доп
            y[ind-1] = final_A[i, 0] * V
    return V, x, y, step


def simplex_add_iteration(step, A, basis_ind, C, n, m):
    # проверка на допустимость
    # if np.any(A[:, 0] < 0):
    #     print("Basis isn't allowable :(")

    # подсчет оценок
    basis_c = C[basis_ind]
    delta = np.array([(np.sum(A[:, j] * basis_c) - C[j]) for j in range(n)])

    # проверка на оптимальность
    if np.all(delta >= 0):
        print("Basis is allowable and optimal now")
        return step, A, basis_ind, delta

    # поиск индексов k и l
    k = np.argmin(delta)

    B_plus_str_ind = []
    for i in range(m):
        if A[i,k] > 0:
            B_plus_str_ind.append(i)
    # if not B_plus_str_ind:
    #     print("F(x) is unlimited from below")

    tetta_0 = [A[ind, 0] / A[ind, k] for ind in B_plus_str_ind]
    l_str_ind = B_plus_str_ind[int(np.argmin(tetta_0))]

    # переход к новому базису
    basis_ind[l_str_ind] = k  # замена l на k
    A[l_str_ind, :] = A[l_str_ind, :] / A[l_str_ind, k]
    for i in range(m):
        if i == l_str_ind:
            continue
        A[i,:] = A[i,:] - A[l_str_ind,:] * A[i,k]

    return simplex_add_iteration(step+1, A, basis_ind, C, n, m)


def simplex_synth_process(A):
    m, n = A.shape
    adapt = np.max(A)
    A = A - adapt
    # все ограничения вида <=0 (для послед = 1)
    b = np.zeros(m+1)
    b[-1] = 1

    A = np.concatenate((A, np.ones((m,1))*-1), axis=1)  # -V1
    A = np.concatenate((A, np.ones((m,1))), axis=1)  # +V2
    # + еще одну строку (условие)
    new_requir = np.concatenate((np.ones((1, n)), np.zeros((1,2))), axis=1)
    A = np.concatenate((A, new_requir), axis=0)
    m_expan = m+1

    # начальный единич базис с пом доп+искусств переменных
    A = np.concatenate((A, np.eye(m+1)), axis=1)
    A = np.concatenate(([np.array(b)], A.T), axis=0).T  # добавили столбец A[0] (BP_0)
    n_expan = A.shape[1]
    basis_index = np.array([i for i in range(n + 3, n_expan)])  # нач базис из доп+искусств переменных

    # коэф-ты целевой функции = 0 (кроме V1, V2, искусств пер)
    C_M, C_num = np.zeros(n_expan), np.zeros(n_expan)
    C_M[-1] = 1  # искусств
    C_num[n+1], C_num[n+2] = 1, -1  # V1, V2

    step, final_A, final_basis_index, final_delta = \
        simplex_synth_iteration(1, A, basis_index, C_M, C_num, n_expan, m_expan)

    V = adapt  # цена игры
    x = final_delta[n_expan-m-1:n_expan-1] * -1
    y = np.zeros(n)
    for i, ind in enumerate(final_basis_index):
        if ind <= n:  # не доп/искусств
            y[ind-1] = final_A[i, 0]
        else:
            V += final_A[i, 0] if ind == n+1 else final_A[i, 0]*-1  # + V1 - V2
    return V, x, y, step


def simplex_synth_iteration(step, A, basis_ind, C_M, C_num, n, m):
    # проверка на допустимость
    # if np.any(A[:, 0] < 0):
    #     print("Basis isn't allowable :(")

    # подсчет оценок
    basis_c_M = C_M[basis_ind]
    delta_M = np.array([(np.sum(A[:, j] * basis_c_M) - C_M[j]) for j in range(n)])
    basis_c_num = C_num[basis_ind]
    delta_num = np.array([(np.sum(A[:, j] * basis_c_num) - C_num[j]) for j in range(n)])

    # проверка на оптимальность
    if np.all(delta_M <= 0) and np.all(delta_num <= 0):
        print("Basis is allowable and optimal now")
        return step, A, basis_ind, delta_num

    # поиск индексов k и l
    k = np.argmax(delta_num[1:] if np.all(delta_M <= 0) else delta_M[1:])
    k+=1
    B_plus_str_ind = []
    for i in range(m):
        if A[i,k] > 0:
            B_plus_str_ind.append(i)
    # if not B_plus_str_ind:
    #     print("F(x) is unlimited from below")

    tetta_0 = [A[ind, 0] / A[ind, k] for ind in B_plus_str_ind]
    l_str_ind = B_plus_str_ind[int(np.argmin(tetta_0))]

    # переход к новому базису
    basis_ind[l_str_ind] = k  # замена l на k
    A[l_str_ind, :] = A[l_str_ind, :] / A[l_str_ind, k]
    for i in range(m):
        if i == l_str_ind:
            continue
        A[i,:] = A[i,:] - A[l_str_ind,:] * A[i,k]

    return simplex_synth_iteration(step+1, A, basis_ind, C_M, C_num, n, m)


if __name__ == '__main__':
    m, n = 0, 0
    sg.theme('LightBrown13')
    font_window = ('Arial 12 bold')
    font_input = ('Arial 12')
    layout_start = [
        [sg.Text('Input num of strategies for 1,2 player:', font=font_window),
         sg.InputText(size=(5,1), font=font_input, key='-num1-'),
         sg.Text(' x '),
         sg.InputText(size=(5,1), font=font_input, key='-num2-')],
        [sg.Button('Ok', font=font_window)]
    ]
    window_start = sg.Window('Shapley vector from KVA', layout_start)

    while True:
        event, values = window_start.read()
        if event in (None, 'Exit'):
            break
        if event == "Ok":
            m = int(values['-num1-'])
            n = int(values['-num2-'])
            break

    layout = [
        [sg.Text('A:')],
        [[sg.Input(size=(6,1), font=font_input, pad=(0,0), key=f'{i}{j}') for i in range(n)] for j in range(m)],
        [sg.Button('Solve', font=font_window)],
        [sg.Text('\n')],
        [sg.Text('Solved using simple way in __ steps', font=font_window, key='-step1-')],
        [sg.Text('V = ', font=font_window, key='-V1-')],
        [sg.Text('x = ', font=font_window, key='-x1-')],
        [sg.Text('y = ', font=font_window, key='-y1-')],
        [sg.Text('Solved using synthetic way in __ steps', font=font_window, key='-step2-')],
        [sg.Text('V = ', font=font_window, key='-V2-')],
        [sg.Text('x = ', font=font_window, key='-x2-')],
        [sg.Text('y = ', font=font_window, key='-y2-')]
    ]
    window = sg.Window('Matr play from KVA', layout)

    while True:
        event, values = window.read()
        if event in (None, 'Exit') or n == 0:
            break

        if event == 'Solve':
            A = np.zeros((m, n))
            is_input_okey = True
            for j in range(m):
                for i in range(n):
                    a_i = values[f'{i}{j}'].replace(' ', '')
                    if not all([a in numbers for a in a_i]):
                        sg.PopupOK('Matr A value can be only number!')
                        is_input_okey = False
                        break
                    A[j,i] = int(a_i)

            if not is_input_okey:
                continue

            V_add, x_add, y_add, step_add = simplex_add_process(A)
            V_synth, x_synth, y_synth, step_synth = simplex_synth_process(A)

            window['-step1-'].update(f'Solved using simple way in {step_add} steps')
            window['-V1-'].update(f'V = {V_add}')
            window['-x1-'].update(f'x = {x_add}')
            window['-y1-'].update(f'y = {y_add}')
            window['-step2-'].update(f'Solved using synthetic'
                                     f' way in {step_synth} steps')
            window['-V2-'].update(f'V = {V_synth}')
            window['-x2-'].update(f'x = {x_synth}')
            window['-y2-'].update(f'y = {y_synth}')
