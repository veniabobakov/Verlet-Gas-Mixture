import numpy as np
import matplotlib.pyplot as plt

# Определяем глобальные физические константы
Avogadro = 6.02214086e23  # Число Авогадро
Boltzmann = 1.38064852e-23  # Постоянная Больцмана


def wallHitCheck(pos, vels, box):
    """Проверяет столкновение частиц со стенками и применяет отражающие граничные условия."""
    ndims = len(box)  # Количество измерений (обычно 3 для 3D)
    for i in range(ndims):
        # Если частица выходит за пределы коробки, меняем направление её скорости
        vels[((pos[:, i] <= box[i][0]) | (pos[:, i] >= box[i][1])), i] *= -1


def integrate(pos, vels, forces, mass, dt):
    """Простой интегратор Эйлера для обновления положения и скорости частиц."""
    pos += vels * dt  # Обновляем положение с учетом текущей скорости
    vels += forces * dt / mass[:, np.newaxis]  # Обновляем скорость с учетом силы и массы


def computeForce(mass, vels, temp, relax, dt):
    """Вычисляет силу для всех частиц с использованием сил Ланжевена."""
    natoms, ndims = vels.shape  # Получаем количество атомов и размерность
    sigma = np.sqrt(2.0 * mass * temp * Boltzmann / (relax * dt))  # Рассчитываем стандартное отклонение для силы
    noise = np.random.randn(natoms, ndims) * sigma[:, np.newaxis]  # Генерируем случайный шум
    force = - (vels * mass[:, np.newaxis]) / relax + noise  # Вычисляем силу
    return force


def generateMaxwellianVelocities(natoms, temp):
    """Генерирует скорости частиц, следуя распределению Максвелла-Больцмана."""
    return np.random.normal(0, np.sqrt(temp * Boltzmann), (natoms, 3))


def computePressure(natoms, temp, volume):
    """Вычисляет давление, используя закон идеального газа."""
    return (natoms * Boltzmann * temp) / volume


def run(**args):
    """Основная функция для запуска симуляции."""
    natoms_left, natoms_right = args['natoms_left'], args[
        'natoms_right']  # Получаем количество атомов в левой и правой половинах
    box, dt = args['box'], args['dt']  # Определяем границы коробки и шаг по времени
    mass, relax, nsteps = args['mass'], args['relax'], args[
        'steps']  # Получаем массу, параметры релаксации и количество шагов

    # Общее количество атомов
    natoms = natoms_left + natoms_right
    dim = len(box)  # Размерность системы (3D)

    # Инициализация положений
    pos = np.zeros((natoms, dim))  # Позиции всех атомов

    # Устанавливаем позиции для левого газа
    for i in range(dim):
        pos[:natoms_left, i] = box[i][0] + (box[i][1] / 2 - box[i][0]) * np.random.rand(natoms_left)

    # Устанавливаем позиции для правого газа
    for i in range(dim):
        pos[natoms_left:, i] = box[i][1] / 2 + (box[i][1] - box[i][1] / 2) * np.random.rand(natoms_right)

    # Генерация начальных скоростей частиц
    vels_left = generateMaxwellianVelocities(natoms_left, args['temp_left'])
    vels_right = generateMaxwellianVelocities(natoms_right, args['temp_right'])
    vels = np.vstack((vels_left, vels_right))  # Объединяем скорости левого и правого газа

    mass = np.ones(natoms) * mass / Avogadro  # Приводим массу к атомной единице
    radius = np.ones(natoms) * args['radius']  # Устанавливаем радиусы частиц
    step = 0  # Счетчик шагов

    output = []  # Для хранения результатов по времени
    pressures = []  # Для хранения давления
    wall_present = True  # Сначала стенка присутствует

    while step <= nsteps:
        step += 1  # Увеличиваем счетчик шагов

        # Удаляем стенку после половины шагов
        if step > nsteps // 2:
            wall_present = False

        # Вычисляем силы
        if wall_present:
            # Применяем силы только к отдельным газам до удаления стенки
            forces_left = computeForce(mass[:natoms_left], vels[:natoms_left], args['temp_left'], relax, dt)
            forces_right = computeForce(mass[natoms_left:], vels[natoms_left:], args['temp_right'], relax, dt)
            forces = np.vstack((forces_left, forces_right))
        else:
            # После удаления стенки используем среднюю температуру для смеси
            temp_mixture = (args['temp_left'] * natoms_left + args['temp_right'] * natoms_right) / natoms
            forces = computeForce(mass, vels, temp_mixture, relax, dt)

        # Перемещаем систему во времени
        integrate(pos, vels, forces, mass, dt)

        # Проверяем столкновения с стенкой
        wallHitCheck(pos, vels, box)

        # Записываем температуру и давление после удаления стенки
        if not wall_present:  # Только записываем после удаления стенки
            ins_temp = np.sum(mass * (vels ** 2).sum(axis=1)) / (Boltzmann * dim * natoms)  # Температура смеси
            output.append([step * dt, ins_temp])  # Записываем время и температуру

            # Вычисляем давление
            ins_pressure = computePressure(natoms, ins_temp, np.prod([box[i][1] - box[i][0] for i in range(dim)]))
            pressures.append(ins_pressure)  # Записываем давление

    return np.array(output), pressures


if __name__ == '__main__':
    params = {
        'natoms_left': 500,  # Количество частиц в левом газе
        'natoms_right': 1000,  # Количество частиц в правом газе
        'mass': 0.001,  # Масса частицы
        'radius': 120e-12,  # Радиус частицы
        'relax': 1e-13,  # Параметр релаксации (влияет на амплитуду шума в силах)
        'dt': 1e-15,  # Шаг по времени
        'steps': 10000,  # Общее количество шагов
        'freq': 100,  # Частота записи данных в файл
        'box': ((0, 1e-8), (0, 1e-8), (0, 1e-8)),  # Границы коробки (размеры)
        'temp_left': 300,  # Температура левого газа
        'temp_right': 100  # Температура правого газа
    }

    output, pressures = run(**params)

    # Преобразуем результаты в массивы NumPy для удобного доступа
    output = np.array(output)

    # Построение графика эволюции температуры после удаления стенки
    plt.figure(figsize=(10, 5))
    plt.plot(output[:, 0] * 1e12, output[:, 1], label="Температура смеси", color='red')

    plt.xlabel('Время (пс)')
    plt.ylabel('Температура (K)')
    plt.title('Эволюция температуры смеси после удаления стенки')
    plt.legend()
    plt.grid()
    plt.show()

    # Построение графика эволюции давления после удаления стенки
    plt.figure(figsize=(10, 5))
    plt.plot(output[:, 0] * 1e12, pressures, label="Давление смеси", color='green')

    plt.xlabel('Время (пс)')
    plt.ylabel('Давление (Па)')
    plt.title('Эволюция давления в смеси после удаления стенки')
    plt.legend()
    plt.grid()
    plt.show()
