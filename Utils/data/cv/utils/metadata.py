def extract_metadata(dataset):
    result = {}

    if hasattr(dataset, "classes"):
        result["class_names"] = dataset.classes
        result["num_classes"] = len(dataset.classes)

    elif hasattr(dataset, "targets"):
        result["num_classes"] = len(set(dataset.targets))

    if hasattr(dataset, "data"):
        data = dataset.data
        shape = data.shape
        if len(shape) == 4:
            result["channels"] = shape[-1]
            result["input_shape"] = (shape[-1], shape[1], shape[2])
        elif len(shape) == 3:
            result["channels"] = 1
            result["input_shape"] = (1, shape[1], shape[2])

    return result
