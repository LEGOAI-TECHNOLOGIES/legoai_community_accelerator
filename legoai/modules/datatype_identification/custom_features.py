# ====================================================================
#  Importing the required python packages
# ====================================================================

import pandas as pd
import numpy as np
from nltk.corpus import words
from trieregex import TrieRegEx as TRE
from functional import pseq

import re
from collections import Counter, OrderedDict
import multiprocessing

from legoai.core.configuration import MODEL_CONFIG

# logger = Logger.getLogger(parent_folder_name = "datatype_l1_identification",child_folder_name="feature")
# ====================================================================
# Creating the words list for spell check or valid text
# Creating the TRIE Regex Branch based on the word list created
# ====================================================================

words_list = []
for value in words.words():
    if len(value) > MODEL_CONFIG['THRESHOLD']['CORPUS_WORD_LIMIT']:
        words_list.append(value.lower())

### TRIE REGEX for the words list
tre = TRE()
tre = TRE(*words_list)

# Get the core count and partition size for multiprocessing
core_count = multiprocessing.cpu_count()  # 1
size = MODEL_CONFIG['THRESHOLD']['PARTITION_SIZE']

# Get the minimum and maximum range for the date extraction
minRange = MODEL_CONFIG['THRESHOLD']['DATE_MIN_YEAR']
maxRange = MODEL_CONFIG['THRESHOLD']['DATE_MAX_YEAR']

# Get the number pattern
NUMBER_PATTERN = re.compile(MODEL_CONFIG['PREPROCESS_CONSTANT']['NUMBER_PATTERN'])


def check_int(strs: str) -> int:
    """
    - Check if the data is of Integer type or not
    - If integer then return 1 else 0

    Parameters
    ----------
    strs(str): input data

    Returns
    -------
    int: flag 0 or 1
    """
    
    ### If integer type then return 1 else return 0
    if isinstance(strs.replace(',', ''), int):
        return 1
    elif isinstance(strs.replace(',', ''), float): ### If the data is matching for float return 0
        return 0
    else:
        try:
            int(strs.replace(',', ''))
            return 1
        except:
            return 0

def check_float(strs) -> int:
    """
    - Check if the data is of Float type or not
    - If float then return 1 else 0

    Parameters
    ----------
    strs (str): input data

    Returns
    -------
    int: flag 0 or 1
    """
    ### If Float type then return else return 0
    if isinstance(strs, float):
        return 1
    else:
        try:
            if check_int(strs.replace(',', '')): ### If the data is matching for Integer return 0
                return 0
            strs = float(strs.replace(',', ''))
            if strs != np.inf:     ### If the data is matching for Infinity return 1
                return 1
            else:
                return 0
        except:
            return 0


def alpha_and_numeric_match(value) -> str:
    """
    - Calculate the ratio of alpha to numeric ratio
    - If the data is only alpha then returns alpha
    - If the data is only numeric then return numeric
    - If the data contains both alpha and numeric or special character then return alphanumeric

    Parameters
    ----------
    value (str): input data

    Returns
    -------
    str: 'alphanumeric' or 'numeric' or 'alphabets' or 'others'
    """
    
    ### Converts the values into string type
    value = str(value)
    
    ### Get the length of 
    charCount = len(re.findall(string=value, pattern=MODEL_CONFIG['PREPROCESS_CONSTANT']['TEXT_PATTERN']))
    numCount = len(re.findall(string=value, pattern=MODEL_CONFIG['PREPROCESS_CONSTANT']['NUMBER_PATTERN']))
    specialCharCount = len(re.findall(string=value, pattern=MODEL_CONFIG['PREPROCESS_CONSTANT']['SPECIAL_CHAR_PATTERN']))

    ### Based on the occurence of the characters and number return the respective types
    if (charCount > 0 or specialCharCount) and numCount > 0:
        return 'alphanumeric'
    elif numCount > 0:
        return 'numeric'
    elif charCount > 0:
        return 'alpha'
    else:
        return 'others'


def check_date(strs: str) -> int:
    """
    - Validate if the input data is of date/datetime format
    - Preprocess and check the string against the pattern and return flag

    Parameters
    ----------
    strs (str): input data

    Returns
    -------
    int: flag 0 or 1

    """
    date_pattern = " \d{4}-[0-1][0-9]-[0-3][0-9](-\d{1,2}:\d{1,2}(:\d{1,2})*(z)*[-\d{0,3}]*(AM|PM|CDT|EDT|IST)*)* | \d{4}-[0-3][0-9]-[0-1][0-9](-\d{1,2}:\d{1,2}(:\d{1,2})*(z)*[-\d{0,3}]*(AM|PM|CDT|EDT|IST)*)* | [0-1][0-9]-\d{4}-[0-3][0-9](-\d{1,2}:\d{1,2}(:\d{1,2})*(z)*[-\d{0,3}]*(AM|PM|CDT|EDT|IST)*)* | [0-3][0-9]-\d{4}-[0-1][0-9](-\d{1,2}:\d{1,2}(:\d{1,2})*(z)*[-\d{0,3}]*(AM|PM|CDT|EDT|IST)*)* | [0-1][0-9]-[0-3][0-9]-\d{4}(-\d{1,2}:\d{1,2}(:\d{1,2})*(z)*[-\d{0,3}]*(AM|PM|CDT|EDT|IST)*)* | [0-3][0-9]-[0-1][0-9]-\d{4}(-\d{1,2}:\d{1,2}(:\d{1,2})*(z)*[-\d{0,3}]*(AM|PM|CDT|EDT|IST)*)* | \d{1,2}-\d{1,2}-\d{4}(-\d{1,2}:\d{1,2}(:\d{1,2})*(z)*[-\d{0,3}]*(AM|PM|CDT|EDT|IST)*)* | \d{4}-\d{1,2}-\d{1,2}(-\d{1,2}:\d{1,2}(:\d{1,2})*(z)*[-\d{0,3}]*(AM|PM|CDT|EDT|IST)*)* | \d{1,2}-\d{4}-\d{1,2}(-\d{1,2}:\d{1,2}(:\d{1,2})*(z)*[-\d{0,3}]*(AM|PM|CDT|EDT|IST)*)* | \d{2,4}-(\\bjanuary\\b|\\bfebruary\\b|\\bmarch\\b|\\bapril\\b|\\bmay\\b|\\bjune\\b|\\bjuly\\b|\\baugust\\b|\\bseptember\\b|\\boctober\\b|\\bnovember\\b|\\bdecember\\b)+-\d{1,2}(-\d{1,2}:\d{1,2}(:\d{1,2})*(z)*[-\d{0,3}]*(AM|PM|CDT|EDT|IST)*)* | \d{2,4}-\d{1,2}-(\\bjanuary\\b|\\bfebruary\\b|\\bmarch\\b|\\bapril\\b|\\bmay\\b|\\bjune\\b|\\bjuly\\b|\\baugust\\b|\\bseptember\\b|\\boctober\\b|\\bnovember\\b|\\bdecember\\b)+(-\d{1,2}:\d{1,2}(:\d{1,2})*(z)*[-\d{0,3}]*(AM|PM|CDT|EDT|IST)*)* | (\\bjanuary\\b|\\bfebruary\\b|\\bmarch\\b|\\bapril\\b|\\bmay\\b|\\bjune\\b|\\bjuly\\b|\\baugust\\b|\\bseptember\\b|\\boctober\\b|\\bnovember\\b|\\bdecember\\b)+-\d{2,4}-\d{1,2}(-\d{1,2}:\d{1,2}(:\d{1,2})*(z)*[-\d{0,3}]*(AM|PM|CDT|EDT|IST)*)*  | (\\bjanuary\\b|\\bfebruary\\b|\\bmarch\\b|\\bapril\\b|\\bmay\\b|\\bjune\\b|\\bjuly\\b|\\baugust\\b|\\bseptember\\b|\\boctober\\b|\\bnovember\\b|\\bdecember\\b)+-\d{1,2}-\d{2,4}(-\d{1,2}:\d{1,2}(:\d{1,2})*(z)*[-\d{0,3}]*(AM|PM|CDT|EDT|IST)*)*  | \d{1,2}-(\\bjanuary\\b|\\bfebruary\\b|\\bmarch\\b|\\bapril\\b|\\bmay\\b|\\bjune\\b|\\bjuly\\b|\\baugust\\b|\\bseptember\\b|\\boctober\\b|\\bnovember\\b|\\bdecember\\b)+-\d{2,4}(-\d{1,2}:\d{1,2}(:\d{1,2})*(z)*[-\d{0,3}]*(AM|PM|CDT|EDT|IST)*)* | \d{1,2}-\d{2,4}-(\\bjanuary\\b|\\bfebruary\\b|\\bmarch\\b|\\bapril\\b|\\bmay\\b|\\bjune\\b|\\bjuly\\b|\\baugust\\b|\\bseptember\\b|\\boctober\\b|\\bnovember\\b|\\bdecember\\b)+(-\d{1,2}:\d{1,2}(:\d{1,2})*(z)*[-\d{0,3}]*(AM|PM|CDT|EDT|IST)*)*  | \d{2,4}-(\\bjan\\b|\\bfeb\\b|\\bmar\\b|\\bapr\\b|\\bmay\\b|\\bjun\\b|\\bjul\\b|\\baug\\b|\\bsep\\b|\\bsept\\b|\\boct\\b|\\bnov\\b|\\bdec\\b)+-\d{1,2}(-\d{1,2}:\d{1,2}(:\d{1,2})*(z)*[-\d{0,3}]*(AM|PM|CDT|EDT|IST)*)*  | \d{2,4}-\d{1,2}-(\\bjan\\b|\\bfeb\\b|\\bmar\\b|\\bapr\\b|\\bmay\\b|\\bjun\\b|\\bjul\\b|\\baug\\b|\\bsep\\b|\\bsept\\b|\\boct\\b|\\bnov\\b|\\bdec\\b)+(-\d{1,2}:\d{1,2}(:\d{1,2})*(z)*[-\d{0,3}]*(AM|PM|CDT|EDT|IST)*)* | (\\bjan\\b|\\bfeb\\b|\\bmar\\b|\\bapr\\b|\\bmay\\b|\\bjun\\b|\\bjul\\b|\\baug\\b|\\bsep\\b|\\bsept\\b|\\boct\\b|\\bnov\\b|\\bdec\\b)+-\d{2,4}-\d{1,2}(-\d{1,2}:\d{1,2}(:\d{1,2})*(z)*[-\d{0,3}]*(AM|PM|CDT|EDT|IST)*)* | (\\bjan\\b|\\bfeb\\b|\\bmar\\b|\\bapr\\b|\\bmay\\b|\\bjun\\b|\\bjul\\b|\\baug\\b|\\bsep\\b|\\bsept\\b|\\boct\\b|\\bnov\\b|\\bdec\\b)+-\d{1,2}-\d{2,4}(-\d{1,2}:\d{1,2}(:\d{1,2})*(z)*[-\d{0,3}]*(AM|PM|CDT|EDT|IST)*)* | \d{1,2}-(\\bjan\\b|\\bfeb\\b|\\bmar\\b|\\bapr\\b|\\bmay\\b|\\bjun\\b|\\bjul\\b|\\baug\\b|\\bsep\\b|\\bsept\\b|\\boct\\b|\\bnov\\b|\\bdec\\b)+-\d{2,4}(-\d{1,2}:\d{1,2}(:\d{1,2})*(z)*[-\d{0,3}]*(AM|PM|CDT|EDT|IST)*)* | \d{1,2}-\d{2,4}-(\\bjan\\b|\\bfeb\\b|\\bmar\\b|\\bapr\\b|\\bmay\\b|\\bjun\\b|\\bjul\\b|\\baug\\b|\\bsep\\b|\\bsept\\b|\\boct\\b|\\bnov\\b|\\bdec\\b)+(-\d{1,2}:\d{1,2}(:\d{1,2})*(z)*[-\d{0,3}]*(AM|PM|CDT|EDT|IST)*)* | \d{2,4}-(\\bjanuary\\b|\\bfebruary\\b|\\bmarch\\b|\\bapril\\b|\\bmay\\b|\\bjune\\b|\\bjuly\\b|\\baugust\\b|\\bseptember\\b|\\boctober\\b|\\bnovember\\b|\\bdecember\\b)+(-\d{1,2}:\d{1,2}(:\d{1,2})*(z)*[-\d{0,3}]*(AM|PM|CDT|EDT|IST)*)* | \d{2,4}-(\\bjan\\b|\\bfeb\\b|\\bmar\\b|\\bapr\\b|\\bmay\\b|\\bjun\\b|\\bjul\\b|\\baug\\b|\\bsept\\b|\\boct\\b|\\bnov\\b|\\bdec\\b)+(-\d{1,2}:\d{1,2}(:\d{1,2})*(z)*[-\d{0,3}]*(AM|PM|CDT|EDT|IST)*)* | (\\bjanuary\\b|\\bfebruary\\b|\\bmarch\\b|\\bapril\\b|\\bmay\\b|\\bjune\\b|\\bjuly\\b|\\baugust\\b|\\bseptember\\b|\\boctober\\b|\\bnovember\\b|\\bdecember\\b)+-\d{2,4} (-\d{1,2}:\d{1,2}(:\d{1,2})*(z)*[-\d{0,3}]*(AM|PM|CDT|EDT|IST)*)* | (\\bjan\\b|\\bfeb\\b|\\bmar\\b|\\bapr\\b|\\bmay\\b|\\bjun\\b|\\bjul\\b|\\baug\\b|\\bsep\\b|\\bsept\\b|\\boct\\b|\\bnov\\b|\\bdec\\b)+-\d{2,4}(-\d{1,2}:\d{1,2}(:\d{1,2})*(z)*[-\d{0,3}]*(AM|PM|CDT|EDT|IST)*)* | \d{1,2}-(\\bjanuary\\b|\\bfebruary\\b|\\bmarch\\b|\\bapril\\b|\\bmay\\b|\\bjune\\b|\\bjuly\\b|\\baugust\\b|\\bseptember\\b|\\boctober\\b|\\bnovember\\b|\\bdecember\\b)+(-\d{1,2}:\d{1,2}(:\d{1,2})*(z)*[-\d{0,3}]*(AM|PM|CDT|EDT|IST)*)* | \d{1,2}-(\\bjan\\b|\\bfeb\\b|\\bmar\\b|\\bapr\\b|\\bmay\\b|\\bjun\\b|\\bjul\\b|\\baug\\b|\\bsep\\b|\\bsept\\b|\\boct\\b|\\bnov\\b|\\bdec\\b)+(-\d{1,2}:\d{1,2}(:\d{1,2})*(z)*[-\d{0,3}]*(AM|PM|CDT|EDT|IST)*)* | (\\bjanuary\\b|\\bfebruary\\b|\\bmarch\\b|\\bapril\\b|\\bmay\\b|\\bjune\\b|\\bjuly\\b|\\baugust\\b|\\bseptember\\b|\\boctober\\b|\\bnovember\\b|\\bdecember\\b)+-\d{1,2}(-\d{1,2}:\d{1,2}(:\d{1,2})*(z)*[-\d{0,3}]*(AM|PM|CDT|EDT|IST)*)* | (\\bjan\\b|\\bfeb\\b|\\bmar\\b|\\bapr\\b|\\bmay\\b|\\bjun\\b|\\bjul\\b|\\baug\\b|\\bsep\\b|\\bsept\\b|\\boct\\b|\\bnov\\b|\\bdec\\b)+-\d{1,2}(-\d{1,2}:\d{1,2}(:\d{1,2})*(z)*[-\d{0,3}]*(AM|PM|CDT|EDT|IST)*)* | \d{2,4}-(\\bjanuary\\b|\\bfebruary\\b|\\bmarch\\b|\\bapril\\b|\\bmay\\b|\\bjune\\b|\\bjuly\\b|\\baugust\\b|\\bseptember\\b|\\boctober\\b|\\bnovember\\b|\\bdecember\\b)+-\d{2,4}(-\d{1,2}:\d{1,2}(:\d{1,2})*(z)*[-\d{0,3}]*(AM|PM|CDT|EDT|IST)*)* | \d{2,4}-(\\bjan\\b|\\bfeb\\b|\\bmar\\b|\\bapr\\b|\\bmay\\b|\\bjun\\b|\\bjul\\b|\\baug\\b|\\bsep\\b|\\bsept\\b|\\boct\\b|\\bnov\\b|\\bdec\\b)+-\d{2,4}(-\d{1,2}:\d{1,2}(:\d{1,2})*(z)*[-\d{0,3}]*(AM|PM|CDT|EDT|IST)*)* | \d{1,2}-days-\d{1,2}:\d{1,2}:\d{1,2}(-\d{1,2}:\d{1,2}(:\d{1,2})*(z)*[-\d{0,3}]*(AM|PM|CDT|EDT|IST)*)* | \d{1,2}h[-]*\d{1,2}m | \d{1,2}:\d{2}[:|-]+\d{2} | \d{1,2}:\d{2}[-]*(AM|PM|CDT|EDT|IST)* | (FY|FQ)+[-]*\d{2,4} | \d+[-]*(year[s]*|month[s]*|day[s]*|week[s]*|year[s]*|hour[s]|minute[s]|second[s])+ | \d{1,2}-\d{2}-\d{2}[-]*(AM|PM|CDT|EDT|IST)* | \d{1,2}:\d{2}-(AM|PM|CDT|EDT|IST)+-\d{1,2}:\d{2}-(AM|PM|CDT|EDT|IST)+ | \b(?:20\d{2}|19\d{2}|\d{2})[-/._](0[1-9]|1[0-2])[-/._](0[1-9]|[12]\d|3[01])T([01]\d|2[0-4])[:,.](0[0-9]|[1-5]\d)[:,.](0[0-9]|[1-5]\d)Z\b | \b(?:20\d{2}|19\d{2}|\d{2})[-/._](0[1-9]|[12]\d|3[01])[-/._](0[1-9]|1[0-2])T([01]\d|2[0-4])[:,.](0[0-9]|[1-5]\d)[:,.](0[0-9]|[1-5]\d)Z\b"
    
    ### Preprocess the text for pattern matching 
    strs = str(strs).replace('/', ' ').replace(',', ' ').replace('.', ' ').replace(' ', '-')
    strs = re.sub(string=strs, pattern='-+', repl='-')
    
    ### Match the text with date pattern and return 1 or 0 based on matching
    matched = re.match(string=" " + strs + " ", pattern=date_pattern, flags=re.I)
    return 1 if matched else 0

def check_other_date(strs) -> int:
    """
    - Match the text with day pattern and return 1 or 0 based on matching
    Parameters
    ----------
    strs (str): input data

    Returns
    -------
    int: flag 0 or 1
    """
    
    ### Pattern for the day checks
    days_abbr = ['saturday', 'sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'october', 'november',
                 'december', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september']
    days_abbr_patt = '\\b' + '\\b|\\b'.join(days_abbr) + '\\b'

    ### Match the text with day pattern and return 1 or 0 based on matching    
    day_check = re.findall(string=strs, pattern=days_abbr_patt, flags=re.I)
    return 1 if len(day_check) > 0 else 0


def check_date_range(strs: str) -> int:
    """
    - Match the date range and return 1 or 0 based on condition
    Parameters
    ----------
    strs (str): string input

    Returns
    -------
    int - boolean check 0 or 1
    """
    try:
        if minRange <= int(strs) <= maxRange:
            return 1
        else:
            return 0
    except:
        return 0


def check_range(vals: str) -> int:
    """
    - Match the range pattern and return the flag based on the match condition

    Parameters
    ----------
    vals (str): string input

    Returns
    -------
    int
    """
    range_data = re.findall(pattern=MODEL_CONFIG['PREPROCESS_CONSTANT']['RANGE_PATTERN'], string=str(vals))
    if len(range_data) > 0 and float(range_data[0][1]) >= float(range_data[0][0]):
        return 1

def get_word_length(val: str) -> int:
    """
    - Check the word length
    Parameters
    ----------
    val (str): input data

    Returns
    -------
    int: length of the word
    """
    return len(val.split(' '))


def lexical_matching(cleaned_data: list) -> float:
    """
    - Check if the cleaned data is empty
    - Code should be calculated at Element level and not at word level
    - Match each text with the TRIE regex on nltk word list
    - Get the total matched text with english dictionary to total data

    Parameters
    ----------
    cleaned_data (list): list of input data

    Returns
    -------
    float: ratio of total matched text in english dictionary to total data
    """
    final_score = []
    
    ### Check if the cleaned data is empty
    if len(cleaned_data) == 0:
        return 0
    else:
        words_freq = Counter(cleaned_data)

        ### Code should be calculated at Element level and not at word level
        for search_word in words_freq.keys():
            bool_score = 0
            for words in search_word.lower().split():
                if tre.has(words):
                    bool_score = 1
                    break
            final_score.append(bool_score * words_freq[search_word])
        
        # Get the total matched text with english dictionary to total data
        return sum(final_score) / len(cleaned_data)


def alphanum_flag_creation(values: list, alpha: int, numeric: int) -> int | float:
    """
    - Change in the feature for alphanumeric features

    Parameters
    ----------
    values (list): list of input data
    alpha (int): alphabet flag
    numeric (int): numeric flag

    Returns
    -------
    int | float: boolean flag or ratio of total matched text in english dictionary to total data
    """
    ### Change in the feature for alphanumeric features
    if alpha == 1:
        return 1
    elif numeric == 1:
        return 0
    else:
        return lexical_matching(values)


def count_pattern_in_cells_with_non_zero_count(values: list, pat: str) -> tuple:
    """
    - Get the length of matching patterns present in the string for each value
    - Count the number of elements with pattern matched in each text

    Parameters
    ----------
    values (list): list of data values
    pat (str): regex pattern for counting within the cells

    Returns
    -------
    tuple: total number of elements with matched pattern, and total cell counts
    """
    cell_counts = [len(re.findall(pat, s)) for s in values]
    return sum(1 for c in cell_counts if c > 0), cell_counts


def l1_level_features(col_values: list, col_values_wo_nan_uncased: list, date_samples: int = 1000) -> list:
    """
    - Subset values for the alphanumeric, alpha, numeric and others type
    - Calculate the upper and lower case characters and ratio and mean values
    - Iterate through each element and get the int, float, range type, date data for feature creation
    - Get the ratio of the various features based on total values
    - Get the statistical values for the Integer, Float, Date type, word length
    - Get the url length and alphanum dictionary feature for each values

    Parameters
    ----------
    col_values (list): list of input data
    col_values_wo_nan_uncased (list): list of data without nan and uncased
    date_samples (int): maximum size of data to extract features ( default: 1000)

    Returns
    -------
    list: list of extracted features dateRatio, wordlen_mean, rangeRatio, floatRatio, zero_flag, intRatio, alphaNumRatio, alphaRatio, frac_unique_sample, flag_numcells
    """
    
    # Define the values for storing the features
    int_type , float_type , alpha_type , alphanum_type , range_type , date_type = [],[],[],[],[],[]
    
    #no of cells with unique contents
    num_unique = len(set(col_values_wo_nan_uncased))

    # initially set all feature values to zero
    dateRatio,wordlen_mean,rangeRatio,floatRatio, zero_flag,intRatio,alphaNumRatio,alphaRatio = 0 , 0 , 0 , 0 , 0 , 0 ,0 , 0
    frac_unique_sample , flag_numcells = 0 , 0
    
    # fraction of cells with unique contents
    frac_unique_sample = 0 if len(col_values_wo_nan_uncased) == 0 else num_unique/len(col_values_wo_nan_uncased)
    
     ### Fraction of cells with numeric content -> frac text cells doesn't add information
    numeric_cell_nz_count, numeric_char_counts = count_pattern_in_cells_with_non_zero_count(
        col_values, NUMBER_PATTERN
    )
    #flag for numeric values
    flag_numcells = np.mean([1 if val > 0 else 0 for val in numeric_char_counts])
    

    
    # subset the values which is not empty for the feature creation    
    col_values = [values for values in col_values if len(values) > 0]
    total_vals = len(col_values)

    # Subset values for the alphanumeric, alpha, numeric and others type
    # logger.info('Custom feature creation alphaAndNumericMatch started:%s', datetime.now())
    alphaNum = pseq(map(alpha_and_numeric_match, col_values), processes=core_count, partition_size=size)
    alphaNum = list(alphaNum)
    alpha_type = list(filter(lambda item: item == 'alpha', alphaNum))
    alphanum_type = list(filter(lambda item: item == 'alphanumeric', alphaNum))

    # Iterate through each elements and get the integer type data for feature creation
    #logger.info('Custom feature creation checkInt started:%s',datetime.now())
    int_data = pseq(map(check_int, col_values), processes=core_count, partition_size=size)
    int_type = [int(col_values[idx].replace(',', '')) for idx, val in enumerate(int_data) if val == 1]
    int_type = list(int_type)

    # Iterate through each elements and get the float type data for feature creation
    #logger.info('Custom feature creation checkFloat started:%s',datetime.now())
    float_data = pseq(map(check_float, col_values), processes=core_count, partition_size=size)
    float_type = [float(col_values[idx].replace(',', '')) for idx, val in enumerate(float_data) if val == 1]
    float_type = list(float_type)

    # Iterate through each elements and get the range type data for feature creation
    #logger.info('Custom feature creation checkRange started:%s', datetime.now())
    range_data = pseq(map(check_range, col_values), processes=core_count, partition_size=size)
    range_type = list(filter(lambda item: item == 1, range_data))
    range_type = list(range_type)

    # Iterate through each elements and get the date type data for feature creation
    #logger.info('Custom feature creation checkDate started:%s', datetime.now())
    sub_values = col_values[:date_samples]

    # Iterate through each elements and get the date type data for feature creation
    date_data = pseq(map(check_date, sub_values), processes=core_count, partition_size=size)
    date_data = list(date_data)

    # Iterate through each elements and get the other date type data for feature creation
    #logger.info('Custom feature creation checkOtherDate started:%s', datetime.now())
    day_data = pseq(map(check_other_date, sub_values), processes=core_count, partition_size=size)
    day_data = list(day_data)
    
    # Iterate through each elements and get the date range data for feature creation
    #logger.info('Custom feature creation DateRange started:%s', datetime.now())
    daterange_data = pseq(map(check_date_range, sub_values), processes=core_count, partition_size=size)
    daterange_data = list(daterange_data)

    date_type = [max(date_data[val], day_data[val], daterange_data[val]) for val in range(len(date_data))]

    # Get the ratio of the various features based on total values
    alphaNumRatio = len(alphanum_type) / total_vals if total_vals > 0 else 0
    alphaRatio = len(alpha_type) / total_vals if total_vals > 0 else 0
    dateRatio = np.mean(date_type) if total_vals > 0 else 0
    intRatio = len(int_type) / total_vals if total_vals > 0 else 0
    floatRatio = len(float_type) / total_vals if total_vals > 0 else 0
    rangeRatio = len(range_type) / total_vals if total_vals > 0 else 0


    #logger.info('Custom feature creation Float features:%s', datetime.now())

    try:
        float_elem = [str(float(val)).split('.') for val in float_type if pd.notnull(val)]
        max_after_float = max([float(val[1]) for val in float_elem if len(val) > 1])

    except Exception as e:
        max_after_float = np.NaN

    ### Get the Float related values before and after float types      
    orig_max_after_float = max_after_float
    max_after_float = max_after_float if pd.notnull(max_after_float) else 0

    if pd.isnull(orig_max_after_float):
        zero_flag = 0
    elif max_after_float > 0:
        zero_flag = 0
    else:
        zero_flag = 1

    # Get the statistical values for the word length 
    #logger.info('Custom feature creation word length features:%s', datetime.now())
    word_len_data = pseq(map(get_word_length, col_values), processes=core_count, partition_size=size)
    word_len_data = list(word_len_data)
    
    if len(word_len_data) > 0:
        wordlen_mean = np.mean(word_len_data)
        
    #logger.info('Custom feature creation case based features:%s', datetime.now())


    return [dateRatio,wordlen_mean,rangeRatio,floatRatio, zero_flag,intRatio,alphaNumRatio,alphaRatio,frac_unique_sample,flag_numcells]


def extract_additional_feats(col_values: list, col_values_wo_nan_uncased: list, features: OrderedDict):
    """
    - Creating of additional/custom features
    - Call the additional features for the input values
    - Iterate through created features and store it in the feature dictionary

    Parameters
    ----------
    col_values (list): list of input data
    col_values_wo_nan_uncased (list): list of input data with nan and uncased
    features (dict): dictionary of features

    Returns
    -------

    """
    ### Creating of additional/custom features

    feats_name =['dateRatio','wordlen_mean','rangeRatio','floatRatio','zero_flag','intRatio',
                 'alphaNumRatio','alphaRatio','frac_unique_sample','flag_numcells']

    ### Call the additional features for the input values
    # start_time = datetime.now()
    #logger.info('Custom feature creation started:%s', start_time)
    l1_feats_list = l1_level_features(col_values,col_values_wo_nan_uncased)
    # end_time = datetime.now()
    #logger.info('Custom feature creation completed:%s', end_time)
    #logger.info('Total time taken:%s', end_time - start_time)

    ### Iterate through created features and store it in the feature dictionary
    for iters, name in enumerate(feats_name):
        features[name] = l1_feats_list[iters]