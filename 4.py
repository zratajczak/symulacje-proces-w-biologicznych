import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def create_grid(rows, cols):  # tworzy siatke dwuwymiarowa 50x50
    grid = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(0)
        grid.append(row)
    return grid


def neighbour_count(grid, x,
                    y):  # definiuje sasiedztwo moorea (8 sasiadow) oraz warunki brzgowe (zawiajnie, czyli komorka z lewej skrajnej strony jest sasiadem komorki z prawej skrajnej strony itd. )
    alive = 0  # poczatkowy licznik zywych komorek
    rows = len(grid)  # liczba wierszy liczona na podstawie podanej siatki
    cols = len(grid[0])  # liczba kolumn jw

    for dx in [-1, 0,
               1]:  # sprawdzamy przesuniecia; dx przesuniecie wiersza (-1 w gore, 1 w dol), dy kolumny (-1 w lewo, 1 w prawo)
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:  # pomijamy samą komórkę, bo nie chcemy jej liczyc do sasiadow
                continue

                # warunki brzegowe, modulo sprawia ze jesli wyjdzie poza granice siatki to wroci od drugiej strony siatki
            nx = (x + dx) % rows  # (x + dx) to wspolrzedne sasiada wiersz
            ny = (y + dy) % cols  # (y + dy) sasiad kolumna

            alive += grid[nx][ny]  # dodajemy sasiada (1 - zywy, 0 - martwy) do calkowitej liczby
    return alive


def transition_rules(grid):
    rows = len(grid)
    cols = len(grid[0])
    new_grid = [row[:] for row in grid]  # tworze kopie starej siatki

    for x in range(rows):  # petla po kazdej komorce spisujaca jej obecny stan
        for y in range(cols):
            alive_neighbors = neighbour_count(grid, x, y)  # ilu zywych sasiadow ma komorka (x,y)
            cell = grid[x][y]  # oznacza stan komorki 1/0

            # zasady zywotnosci komorek
            if cell == 1:  # komórka żywa
                if alive_neighbors < 2 or alive_neighbors > 3:
                    new_grid[x][y] = 0  # komorka umiera
                else:
                    new_grid[x][y] = 1  # komorka przeżywa
            else:  # komórka martwa
                if alive_neighbors == 3:
                    new_grid[x][y] = 1  # komorka ożywa

    return new_grid  # nowa siatka/stan


def count_alive_cells(grid):  # liczy liczbę zywych komorek
    return sum(sum(row) for row in grid)


def insert_pattern(grid, pattern, x,
                   y):  # wzorce ciaglego zycia lub oscylatory - wybieramy "forme" komorek ktora chcemy nalozyc na siatke - w jednej funkcji bo są to dość proste formy i tak wygodniej
    if pattern == "block":
        grid[x][y] = 1
        grid[x][y + 1] = 1
        grid[x + 1][y] = 1
        grid[x + 1][y + 1] = 1

    elif pattern == "beehive":
        grid[x][y + 1] = 1
        grid[x][y + 2] = 1
        grid[x + 1][y] = 1
        grid[x + 1][y + 3] = 1
        grid[x + 2][y + 1] = 1
        grid[x + 2][y + 2] = 1

    elif pattern == "boat":
        grid[x][y] = 1
        grid[x][y + 1] = 1
        grid[x + 1][y] = 1
        grid[x + 1][y + 2] = 1
        grid[x + 2][y + 1] = 1

    # oscylatory
    elif pattern == "blinker":
        grid[x][y] = 1
        grid[x][y + 1] = 1
        grid[x][y + 2] = 1

    elif pattern == "toad":
        grid[x][y + 1] = 1
        grid[x][y + 2] = 1
        grid[x][y + 3] = 1
        grid[x + 1][y] = 1
        grid[x + 1][y + 1] = 1
        grid[x + 1][y + 2] = 1

    else:
        raise ValueError(f"Nieznany wzorzec: {pattern}")


def still_life(grid):  # wykonanie pkt 2 - symulacja wzorcow ciaglego zycia
    insert_pattern(grid, "block", 5, 5)
    insert_pattern(grid, "beehive", 20, 20)
    insert_pattern(grid, "boat", 35, 35)


def oscylator(grid):  # wykonanie pkt 3 - symulacja oscylatorow
    insert_pattern(grid, "blinker", 5, 5)
    insert_pattern(grid, "toad", 20, 20)


def glider_gun(grid, x, y):  # wykonanie pkt 4 - działo
    coords = [
        (0, 24),
        (1, 22), (1, 24),
        (2, 12), (2, 13), (2, 20), (2, 21), (2, 34), (2, 35),
        (3, 11), (3, 15), (3, 20), (3, 21), (3, 34), (3, 35),
        (4, 0), (4, 1), (4, 10), (4, 16), (4, 20), (4, 21),
        (5, 0), (5, 1), (5, 10), (5, 14), (5, 16), (5, 17), (5, 22), (5, 24),
        (6, 10), (6, 16), (6, 24),
        (7, 11), (7, 15),
        (8, 12), (8, 13)
    ]
    for dx, dy in coords:
        grid[x + dx][y + dy] = 1


def matuzalek(grid, x, y):  # wykonanie pkt 5 - matuzalek
    coords = [(0, 1), (0, 2), (1, 0), (1, 1), (2, 1)]
    for dx, dy in coords:
        grid[x + dx][y + dy] = 1


def animate(grid, steps, interval=200):  # animacja komorek w kazdej iteracji
    fig, ax = plt.subplots()
    img = ax.imshow(grid, cmap='binary')

    text = ax.text(0.95, 0.95, '', transform=ax.transAxes,
                   fontsize=12, color='red', ha='right', va='top',
                   bbox=dict(facecolor='white', alpha=0.7))

    def update(frame):
        nonlocal grid
        new_grid = transition_rules(grid)

        grid = new_grid
        img.set_data(grid)

        alive_cells = count_alive_cells(grid)
        ax.set_title(f"Epoka: {frame + 1}")  # informacja ktora to epoka oraz ilość zywych komorek w niej
        text.set_text(f"Liczba żywych komórek: {alive_cells}")

        if frame + 1 == steps:
            print(f"Końcowa liczba żywych komórek: {alive_cells}")
            ani.event_source.stop()

        return [img, text]

    ani = animation.FuncAnimation(fig, update, frames=steps, interval=interval, blit=False)
    plt.show()


def main():
    while True:
        mode = input(
            "Wybierz symulację ('1' dla symulacji wzorców ciągłego życia, '2' dla symulacji oscylatorów, '3' dla symulacji wybranego działa, '4' dla symulacji Matuzalecha lub 'end' aby zakończyć): ").strip().lower()
        if mode == "end":
            print("Koniec programu.")
            break

        grid = create_grid(50, 50)

        if mode == "1":
            print("Symulacja wzorców ciągłego życia:")
            still_life(grid)
            print(f"Początkowa liczba żywych komórek: {count_alive_cells(grid)}")
            animate(grid, steps=5)

        elif mode == "2":
            print("Symulacja oscylatorów:")
            oscylator(grid)
            print(f"Początkowa liczba żywych komórek: {count_alive_cells(grid)}")
            animate(grid, steps=10)

        elif mode == "3":
            print("Symulacja działa:")
            glider_gun(grid, 10, 10)
            print(f"Początkowa liczba żywych komórek: {count_alive_cells(grid)}")
            animate(grid, steps=100)

        elif mode == "4":
            print("Symulacja Matuzalecha:")
            matuzalek(grid, 10, 10)
            print(f"Początkowa liczba żywych komórek: {count_alive_cells(grid)}")
            animate(grid, steps=250)

        else:
            print("Błąd: nieznany tryb symulacji")


if __name__ == "__main__":
    main()
