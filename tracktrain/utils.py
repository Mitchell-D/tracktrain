def validate_keys(mandatory_keys:list, received_keys:list, source_name:str="",
                  descriptions:dict={}):
    """
    Raises a descriptive ValueError if a subset of the provided list of keys
    aren't in recieved_keys.

    :@param mandatory_keys: list of keys that must appear in received_keys
    :@param received_keys:  list of keys that were provided by the user
    :@param source_name: name of pipeline configured by the keys
        (ie "train", "compile")
    :@param descriptions: dict mapping at least some of the keys to
        descriptive strings to guide the user in providing an appropriate
        parameter as an argument.
    """
    ## (bool, str) 2-tuple of mandatory args and whether they are provided
    args = [(a in received_keys, a) for a in mandatory_keys]
    excluded_args = list(filter(lambda a:not a[0], args))
    if len(excluded_args)>0:
        ## Add a space if a source name was provided
        if source_name != "":
            source_name = f"'{source_name}' "
        err_str = f"{source_name}config missing mandatory fields:\n"
        missing_fields = tuple(zip(*excluded_args))[1]
        ## If descriptions are provided, print an error message including the
        ## description of each missing parameter having one.
        if not descriptions:
            err_str += ", ".join(missing_fields)
        else:
            descs = [(
                f, f"    {f}:{descriptions.get(f)}"
                )[f in descriptions.keys()]
                     for f in missing_fields]
            err_str += "\n".join(descs)
        raise ValueError(err_str)
    return True

if __name__=="__main__":
    assert validate_keys(
            mandatory_keys=("a", "b", "c"),
            received_keys=("a", "b", "c", "d", "e"),
            source_name=("test1"),
            )

    try:
        validate_keys(
                mandatory_keys=("a", "b", "c"),
                received_keys=("a", ),
                source_name=("test2"),
                descriptions={
                    "a":"The first argument",
                    "b":"The second argument",
                    "c":"The third argument",
                    }
                )
    except ValueError as e:
        print(f"Test failed successfully:")
        print(e)

