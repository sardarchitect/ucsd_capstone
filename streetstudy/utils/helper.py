def dict_key_string_to_int(dictionary):
    # Converts all dict keys from str to int
    dictionary= {int(k):v for k, v in dictionary.items()}
    return dictionary