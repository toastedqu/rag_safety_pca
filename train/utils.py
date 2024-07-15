def find_best_hyperparameters(results: dict) -> list:
    """
    Find the best hyperparameters from the results
    :param results: dict
        The results of the hyperparameter search
    :return: list
        The best hyperparameters
    """
    best_params = max(results, key=lambda key: results[key])

    if isinstance(best_params, tuple) or isinstance(best_params, list):
        return list(best_params)
    else:
        return [best_params]
