def fmt_time(seconds: int, degrees=2) -> str:
    minutes = seconds // 60
    seconds = seconds % 60

    hours = minutes // 60
    minutes = minutes % 60

    days = hours // 24
    hours = hours % 24

    parts = []
    if days:
        parts.append(f'{days}d')
    if hours:
        parts.append(f'{hours}h')
    if minutes:
        parts.append(f'{minutes}m')
    parts.append(f'{seconds}s')

    return ' '.join(parts[:degrees])
