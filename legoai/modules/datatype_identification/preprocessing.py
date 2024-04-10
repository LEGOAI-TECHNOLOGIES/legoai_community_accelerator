# ====================================================================
#  Importing the required python packages
# ====================================================================

import pandas as pd

import re
import string

# Ignore list for the preprocessing values
ignoreList = ['#na', '#n/a', 'na', 'n/a', 'none', 'nan', 'blank', 'blanks', 'nil', 'n.a.', 'n.a',
              '"#na"', '"#n/a"', '"na"', '"n/a"', '"none"', '"nan"', '"blank"', '"blanks"', '"nil"', '"n.a."', '"n.a"',
              "'#na'", "'#n/a'", "'na'", "'n/a'", "'none'", "'nan'", "'blank'", "'blanks'", "'nil'", "'n.a.'", "'n.a'"]


def remove_non_ascii(strs) -> str:
    """
    - Remove Non ASCII characters from the data
    Parameters
    ----------
    strs (str): input data

    Returns
    -------
    non ascii characters removed data
    """
    return ''.join([char for word in str(strs) for char in word if ord(char) < 128])


def remove_punctuation(strs) -> str:
    """
     -  Remove punctuation characters at start and end of the data

    Parameters
    ----------
    strs (str): input string

    Returns
    -------
    punctuation removed string
    """
    return strs.strip("'").strip('"')


def remove_punctuation_text(strs: str) -> str:
    """
    - Remove punctuation from the text and check if they are empty
    - If it contains only punctuations, then return empty else return string
    Parameters
    ----------
    strs (str): input string

    Returns
    -------
    punctuation removed string
    """
    clean_str = strs.translate(str.maketrans('', '', string.punctuation))
    clean_str = normalise_whitespace(clean_str)
    clean_str = clean_str.strip()
    return '' if len(clean_str) == 0 else strs  ## Checks if the string is empty else return string


def special_token_repl(text: str, suffix: str) -> str:
    """
    - Remove punctuation characters at start and end of the data

    Parameters
    ----------
    text (str): input string
    suffix (str): additional text to be added at the end

    Returns
    -------
    processed input string
    """
    
    ### Replace the pattern with empty space and replace multiple space with single space
    replaced_text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    replaced_text = re.sub(string=replaced_text, pattern=' +', repl=' ')

    ### Check if the replaced text is empty then add unknown and suffix word
    if replaced_text == '':
        replaced_text = 'unknown' + suffix

    return replaced_text


def remove_table_column_name(values: list, dataset_name: str, table_name: str, column_name: str):
    """
    - Check if column values contains dataset , table or column name if not returns the value

    Parameters
    ----------
    values (list): list of column values
    dataset_name (str): name of the dataset
    table_name (str): name of the table
    column_name (str): name of the column

    Returns
    -------
    filtered values
    """
    return [val for val in values if
            str(val).lower() not in [dataset_name.lower(), table_name.lower(), column_name.lower()]]


def additional_processing(value) -> str:
    """
    - Remove null, nan, none and ignore list elements from data
    - Remove ASCII Characters and unicode characters
    - Remove punctuation text and remove punctuation in the start and end text

    Parameters
    ----------
    value (list): list of data

    Returns
    -------
    null, nan, ascii, punctuation remove data
    """

    ### Remove the nulls/none and not in ignore list
    if value is None or pd.isnull(value) or str(value).lower() in ignoreList:
        return_val = ''
    else:
        value = str(value).replace('\xa0', ' ').strip()
        return_val = remove_non_ascii(value)

    ### Remove punctuation in the start and end text
    return_val = remove_punctuation_text(return_val)
    
    ### Remove punctuation text from text
    return_val = remove_punctuation(return_val)
    return return_val


def normalise_whitespace(data) -> str:
    """
    - Clean whitespace from strings by:
    - trimming leading and trailing whitespace
    - normalising all whitespace to spaces
    - reducing whitespace sequences to a single space

    Parameters
    ----------
    data ( str | any): input data

    Returns
    -------
    whitespace normalised datas
    """
    if isinstance(data, str):
        return re.sub(r"\s{2,}", " ", data.strip())
    else:
        return data


def normalise_string_whitespace(col_values) -> list:
    """
    - Clean whitespace from strings by:
    - trimming leading and trailing whitespace
    - normalising all whitespace to spaces
    - reducing whitespace sequences to a single space

    Parameters
    ----------
    col_values (list): list of input data

    Returns
    -------
    list of cleaned values
    """
    
    ### Get the metadata info such as id, dataset, table, column name
    master_id = col_values[0]
    id = col_values[1]
    dataset_name = col_values[2]
    table_name = col_values[3]
    column_name = col_values[4]

    ### Remove the whitespaces from the data
    normalized_values = list(map(normalise_whitespace, col_values[5:]))

    ### Removing the table and column name from values ## Added to remove features list
    normalized_values = [val for val in normalized_values if
                         str(val).lower() not in [dataset_name.lower(), table_name.lower(), column_name.lower()]]
    
    ### Combining the metadata with the normalized values into a list of data
    normalized_values_upd = [master_id] + [id] + [dataset_name] + [table_name] + [column_name] + normalized_values
    return normalized_values_upd


def data_standardization(df: pd.DataFrame) -> pd.DataFrame:
    """
    - setting all string values to lowercase
    - removing leading and trailing whitespace from string values
    - removing leading and trailing non-word characters (such as punctuation) from string values
    
    Parameters
    ----------
    df (pd.DataFrame): The DataFrame to standardize
    
    Returns
    -------
    The standardized DataFrame
    """
    
    # setting case to lower for string rows
    df = df.map(lambda x: x.lower() if isinstance(x, str) else x)
    df = df.map(lambda s: re.sub(r'^\W+|\W+$', '', s.strip()) if isinstance(s, str) else s)
    return df
