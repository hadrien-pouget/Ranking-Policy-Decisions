def sec_to_str(s, padh=2):
    s = int(s)

    h = int(s / 3600)
    s = s % 3600

    m = int(s / 60)
    s = s % 60

    return "{0:0{3}d}:{1:02d}.{2:02d}".format(h, m, s, padh)

def str_to_sec(s):
    h, rest = s.split(':')
    m, s = rest.split('.')
    h, m, s = int(h), int(m), int(s)
    s = s + 60*m + 3600*h
    return s

def get_counting_time(logger):
    """ Get time spent counting from logger """
    return logger.data['logs'][0]['counting_time']
