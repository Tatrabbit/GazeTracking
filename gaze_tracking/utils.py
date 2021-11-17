def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

def inverse_lerp(a, b, value):
    if a != b:
        return clamp((value - a) / float(b - a), 0.0, 1.0)
    return 0.0