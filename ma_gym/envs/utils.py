from PIL import Image, ImageDraw


def draw_grid(rows, cols, cell_size=50, fill='black'):
    height = rows * cell_size
    width = cols * cell_size
    image = Image.new(mode='RGB', size=(width, height), color=fill)

    # Draw some lines
    draw = ImageDraw.Draw(image)
    y_start = 0
    y_end = image.height
    step_size = cell_size

    for x in range(0, image.width, step_size):
        line = ((x, y_start), (x, y_end))
        draw.line(line, fill='black')

    x_start = 0
    x_end = image.width

    for y in range(0, image.height, step_size):
        line = ((x_start, y), (x_end, y))
        draw.line(line, fill='black')

    del draw

    return image


def fill_cell(image, pos, cell_size=50, fill='black'):
    col, row = pos
    row, col = row * cell_size, col * cell_size

    ImageDraw.Draw(image).rectangle([(row, col), (row + cell_size, col + cell_size)], fill=fill)


def draw_cell_outline(image, pos, cell_size=50, fill='black'):
    col, row = pos
    row, col = row * cell_size, col * cell_size

    ImageDraw.Draw(image).rectangle([(row, col), (row + cell_size, col + cell_size)], outline=fill)


def draw_circle(image, pos, cell_size=50, fill='black'):
    col, row = pos
    row, col = row * cell_size, col * cell_size
    gap = cell_size * 0.4
    x, y = row + gap, col + gap
    x_dash, y_dash = row + cell_size - gap, col + cell_size - gap
    ImageDraw.Draw(image).arc([(x, y), (x_dash, y_dash)], start=0, end=360, fill=fill)
