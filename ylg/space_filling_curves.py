class SpaceFillingCurve:
    def enumerate_cells(*args, **kwargs):
        raise NotImplementedError()


class Manhattan(SpaceFillingCurve):
    def enumerate_cells(rows, cols):
        # maps distances to cells
        distances = defaultdict(list)

        # maps numbers to cells
        enumeration = {}

        for i in range(rows):
            for j in range(cols):
                distance = i + j
                distances[distance].append([i, j])

        sorted_distances = sorted(list(distances.keys()))

        numbers = list(range(rows * cols))
        for distance in sorted_distances:
            cells = distances[distance]
            for cell in cells:
                enumeration[numbers.pop(0)] = cell
        return enumeration
