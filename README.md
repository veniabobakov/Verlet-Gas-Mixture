
# Описание программы молекулярной динамики (MD)

Программа моделирует молекулярную динамику двух газов, находящихся в одном объёме, с начальной стенкой, которая исчезает после половины времени симуляции. Используется метод Эйлера для интеграции движения частиц.

## ВВедение 
В данной программе моделируется газ с силой Ленжевина. В одинаковый объем помещаются два газа, разделенные стенкой. Через какое-то время стенка удаляется и подсчитываются общие температуры и давления смеси по следующим формулам:
Давление смеси газа можно вычислить по уравнению состояния идеального газа:
### Давление смеси
\[
P = \frac{N \cdot k_B \cdot T}{V}
\]

где:
- \( P \) — давление газа (Па)
- \( N \) — общее количество частиц (безразмерная величина)
- \( k_B \) — постоянная Больцмана (\(1.38064852 \times 10^{-23} \, \text{Дж/К}\))
- \( T \) — температура газа (К)
- \( V \) — объем газа (м³)

### Температура смеси

Температура смеси может быть вычислена как средняя температура двух газов, взятых по количеству частиц:

\[
T_{mix} = \frac{N_{left} \cdot T_{left} + N_{right} \cdot T_{right}}{N_{left} + N_{right}}
\]

где:
- \( T_{mix} \) — температура смеси (К)
- \( N_{left} \) — количество частиц в левом газе (безразмерная величина)
- \( T_{left} \) — температура левого газа (К)
- \( N_{right} \) — количество частиц в правом газе (безразмерная величина)
- \( T_{right} \) — температура правого газа (К)
## Структура программы

### Импорт библиотек

```python
import numpy as np
import matplotlib.pyplot as plt
```

Импорт необходимых библиотек: NumPy для численных вычислений, Matplotlib для визуализации и модуля dump для записи выходных данных.

Определение физических констант
python
Копировать код
# Определяем глобальные физические константы
Avogadro = 6.02214086e23
Boltzmann = 1.38064852e-23
Здесь определяются числа Авогадро и постоянная Больцмана, которые используются в расчетах.

Функция wallHitCheck
```python
def wallHitCheck(pos, vels, box):
    """Проверяет столкновения с границами и отражает частицы."""
    ndims = len(box)
    for i in range(ndims):
        vels[((pos[:, i] <= box[i][0]) | (pos[:, i] >= box[i][1])), i] *= -1
 ```
Эта функция проверяет, пересекают ли частицы границы коробки. Если частица выходит за пределы, её скорость инвертируется.

Функция integrate
```python
def integrate(pos, vels, forces, mass, dt):
    """Интегрирует движение частиц методом Эйлера."""
    pos += vels * dt
    vels += forces * dt / mass[np.newaxis].T
```

Метод Эйлера используется для обновления положения и скорости частиц. Он основан на простом приближении, где новое положение рассчитывается на основе текущей скорости и силы.

Функция computeForce
```python
def computeForce(mass, vels, temp, relax, dt):
    """Вычисляет силы для всех частиц."""
    natoms, ndims = vels.shape
    sigma = np.sqrt(2.0 * mass * temp * Boltzmann / (relax * dt))
    noise = np.random.randn(natoms, ndims) * sigma[np.newaxis].T
    force = - (vels * mass[np.newaxis].T) / relax + noise
    return force
```

Эта функция вычисляет Langevin силы, учитывающие термальное движение и случайный шум. Параметр relax отвечает за время релаксации системы, определяя, насколько быстро частицы реагируют на термические колебания.

Функция generateMaxwellianVelocities
```python
def generateMaxwellianVelocities(natoms, temp):
    """Генерирует скорости, следуя распределению Максвелла-Больцмана."""
    return np.random.normal(0, np.sqrt(temp * Boltzmann), (natoms, 3))
```
Функция генерирует начальные скорости частиц на основе распределения Максвелла-Больцмана для заданной температуры.

Функция computePressure
```python
def computePressure(natoms, temp, volume):
    """Вычисляет давление с использованием уравнения состояния идеального газа."""
    pressure = (natoms * Boltzmann * temp) / volume
    return pressure
```

Эта функция рассчитывает давление в системе, используя уравнение состояния идеального газа.

Главная функция run
```python
def run(**args):
    """Основная функция для запуска симуляции."""
    natoms_left, natoms_right = args['natoms_left'], args['natoms_right']
    box, dt = args['box'], args['dt']
    mass, relax, nsteps = args['mass'], args['relax'], args['steps']
    
    # Общее количество атомов
    natoms = natoms_left + natoms_right
    dim = len(box)
    
    # Инициализация позиций
    pos = np.zeros((natoms, dim))
    
    # Позиции газа слева
    for i in range(dim):
        pos[:natoms_left, i] = box[i][0] + (box[i][1] / 2 - box[i][0]) * np.random.rand(natoms_left)

    # Позиции газа справа
    for i in range(dim):
        pos[natoms_left:, i] = box[i][1] / 2 + (box[i][1] - box[i][1] / 2) * np.random.rand(natoms_right)

    # Генерация скоростей Максвелла
    vels_left = generateMaxwellianVelocities(natoms_left, args['temp_left'])
    vels_right = generateMaxwellianVelocities(natoms_right, args['temp_right'])
    vels = np.vstack((vels_left, vels_right))

    mass = np.ones(natoms) * mass / Avogadro
    radius = np.ones(natoms) * args['radius']
    step = 0

    output = []
    pressures = []
    wall_present = True

    # Переменные для хранения температур и давлений перед смешиванием
    volume_left = np.prod([box[i][1] / 2 - box[i][0] for i in range(dim)])  # Объем газа слева
    volume_right = np.prod([box[i][1] - box[i][1] / 2 for i in range(dim)])  # Объем газа справа

    while step <= nsteps:
        step += 1

        # Удаляем стенку после половины шагов
        if step > nsteps // 2:
            wall_present = False

        # Вычисление всех сил
        if wall_present:
            forces_left = computeForce(mass[:natoms_left], vels[:natoms_left], args['temp_left'], relax, dt)
            forces_right = computeForce(mass[natoms_left:], vels[natoms_left:], args['temp_right'], relax, dt)
            forces = np.vstack((forces_left, forces_right))
        else:
            # Используем среднюю температуру после удаления стенки для смеси
            temp_mixture = (args['temp_left'] * natoms_left + args['temp_right'] * natoms_right) / natoms
            forces = computeForce(mass, vels, temp_mixture, relax, dt)

        # Перемещение системы во времени
        integrate(pos, vels, forces, mass, dt)

        # Проверка на столкновения со стенками
        wallHitCheck(pos, vels, box)

        # Вычисление давления
        ins_pressure = computePressure(natoms, temp_mixture, np.prod([box[i][1] - box[i][0] for i in range(dim)]))
        pressures.append(ins_pressure)

    return np.array(output), pressures
```

Эта функция запускает основную симуляцию, инициализируя частицы, вычисляя силы и обновляя их положения и скорости. Она также управляет удалением стенки и сохраняет данные о давлении.

Пример запуска программы
```python
if __name__ == '__main__':
    params = {
        'natoms_left': 500,  # Количество частиц в левом газе
        'natoms_right': 500,  # Количество частиц в правом газе
        'mass': 0.001,
        'radius': 120e-12,
        'relax': 1e-13,
        'dt': 1e-15,
        'steps': 10000,
        'freq': 100,
        'box': ((0, 1e-8), (0, 1e-8), (0, 1e-8)),
        'ofname': 'traj-hydrogen-3D.dump',
        'temp_left': 300,  # Температура левого газа
        'temp_right': 100   # Температура правого газа
    }

    output, pressures = run(**params)
```

Этот блок кода задаёт параметры симуляции и запускает основную функцию run, которая выполняет моделирование.