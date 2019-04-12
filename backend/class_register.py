AVAILABLE_MODELS = {}


def register_api(name):
    def decorator(cls):
        print("Found API {} with name {}".format(
            cls, name))
        AVAILABLE_MODELS[name] = cls
        return cls
    return decorator
