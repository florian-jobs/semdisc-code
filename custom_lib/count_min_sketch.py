from probables import CountMinSketch

def build_count_min_sketch(values, width=2000, depth=7):
    cms = CountMinSketch(width=width, depth=depth)
    for value in values:
        cms.add(str(value))
    return cms

def add(cms, value, count=1):
    cms.add(str(value), num_els=count)

def add_to_count_min_sketch(cms, value, frequency=1):
    cms.add(str(value), num_els=frequency)

def get_frequency(cms, value):
    return cms.check(str(value))
