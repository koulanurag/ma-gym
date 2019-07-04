from PIL import Image, ImageDraw


def draw_grid(rows, cols, cell_size=50, fill='black',line_color='black'):
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
        draw.line(line, fill=line_color)

    x_start = 0
    x_end = image.width

    for y in range(0, image.height, step_size):
        line = ((x_start, y), (x_end, y))
        draw.line(line, fill=line_color)

    del draw

    return image


def fill_cell(image, pos, cell_size=None, fill='black', margin=0):
    assert cell_size is not None and 0 <= margin <= 1

    col, row = pos
    row, col = row * cell_size, col * cell_size
    margin *= cell_size
    x, y, x_dash, y_dash = row + margin, col + margin, row + cell_size - margin, col + cell_size - margin
    ImageDraw.Draw(image).rectangle([(x, y), (x_dash, y_dash)], fill=fill)


def draw_cell_outline(image, pos, cell_size=50, fill='black'):
    col, row = pos
    row, col = row * cell_size, col * cell_size
    ImageDraw.Draw(image).rectangle([(row, col), (row + cell_size, col + cell_size)], outline=fill, width=3)


def draw_circle(image, pos, cell_size=50, fill='black'):
    col, row = pos
    row, col = row * cell_size, col * cell_size
    gap = cell_size * 0.4
    x, y = row + gap, col + gap
    x_dash, y_dash = row + cell_size - gap, col + cell_size - gap
    ImageDraw.Draw(image).ellipse([(x, y), (x_dash, y_dash)], outline=fill, fill=fill)
