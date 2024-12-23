def fn_and(x1, x2):
    """
    Функція для перевірки одночасної істинності двох значень.
    Повертає 1, якщо обидва аргументи дорівнюють 1; інакше 0.
    """
    return int(x1 == 1 and x2 == 1)


def fn_or(x1, x2):
    """
    Функція для перевірки істинності хоча б одного аргументу.
    Повертає 1, якщо хоча б один з аргументів дорівнює 1; інакше 0.
    """
    return int(x1 == 1 or x2 == 1)


def fn_xor(x1, x2):
    """
    Обчислення виключного АБО (XOR) з використанням OR і AND.
    Логіка: результат істинний, якщо рівно один з аргументів дорівнює 1.
    """
    or_result = fn_or(x1, x2)
    and_result = fn_and(x1, x2)
    return fn_and(or_result, not and_result)


def test_functions():
    """
    Перевірка правильності роботи логічних функцій на всіх комбінаціях.
    Виводить таблицю істинності для AND, OR і XOR.
    """
    inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    print("x1 | x2 | AND | OR  | XOR")
    print("---|----|-----|-----|-----")
    for x1, x2 in inputs:
        print(f" {x1} |  {x2} |  {fn_and(x1, x2)}  |  {fn_or(x1, x2)}  |  {fn_xor(x1, x2)}")


if __name__ == "__main__":
    test_functions()
