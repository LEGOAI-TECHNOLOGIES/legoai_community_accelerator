# ====================================================================
#  Importing the required python packages
# ====================================================================
import pandas as pd
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

from legoai_di.core.configuration import MODEL_CONFIG


########## Datetype format identifier #############

class L2Model:
    """
    L2 model is mainly responsible for classifying:

    - the integer and float datatype into integer_measure or
    - integer_dimension and same for float,
    - this model also finds the specific format the date & time datatype lies in (e.g. YYYY:MM:DD , YYYY-MM-DD, YYYY:MM:DD H:m:s , etc...)
    - this model uses llm for dimension & measure classification and a regex based approach for the date & time classification.
    """
    def __init__(self, openai_api_key: str, temperature: float = 0.0):
        self.chat_llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=temperature)
        self.dm_class_schema = ResponseSchema(name="DM_class", description="This final class of a column Dimension, Measure & Unknown")
  
    @staticmethod
    def classify_datetime_format(column_values: list, num_samples: int) -> list:
        """
        - Classify the datetime format of a given list of column values.

        Parameters
        ----------
        column_values (list): List of values from a column.
        num_samples (int): Number of values to sample for classification.

        Returns
        -------
        The majority datetime format group.
    """
    
    # Convert non-string items to strings
        column_values = [str(item) if not isinstance(item, str) else item for item in column_values]

    # Sample all column_values if the num_samples exceed total number of column_values
        if len(column_values) > num_samples:
            sampled_values = pd.Series(column_values).sample(n=num_samples, random_state=42).tolist()
        else:
            sampled_values = column_values
            print(f"All values are sampled since number of samples exceed number of column values. \n")

    # Define a mapping of date-time format groups and their patterns
        date_time_groups_new = {
            "YYYY-MM-DD": r"\b(?:20\d{2}|19\d{2}|\d{2})[-./_](0[1-9]|1[0-2])[-./_](0[1-9]|[12]\d|3[01])\b",
            "YYYY-DD-MM": r"\b(?:20\d{2}|19\d{2}|\d{2})[-./_](0[1-9]|[12]\d|3[01])[-./_](0[1-9]|1[0-2])\b",

            "MM-DD-YYYY": r"\b(0[1-9]|1[0-2])[-./_](0[1-9]|[12]\d|3[01])[-./_](?:20\d{2}|19\d{2}|\d{2})\b",
            "DD-MM-YYYY": r"\b(0[1-9]|[12]\d|3[01])[-./_](0[1-9]|1[0-2])[-./_](?:20\d{2}|19\d{2}|\d{2})\b",

            "YYYY-MM-DDTHH:MM:SS": r"\b(?:20\d{2}|19\d{2}|\d{2})[-/._](0[1-9]|1[0-2])[-/._](0[1-9]|[12]\d|3[01])T([01]\d|2[0-4])[:,.](0[0-9]|[1-5]\d)[:,.](0[0-9]|[1-5]\d)\b",
            "YYYY-DD-MMTHH:MM:SS": r"\b(?:20\d{2}|19\d{2}|\d{2})[-/._](0[1-9]|[12]\d|3[01])[-/._](0[1-9]|1[0-2])T([01]\d|2[0-4])[:,.](0[0-9]|[1-5]\d)[:,.](0[0-9]|[1-5]\d)\b",

            "YYYY-MM-DDTHH:MM:SSZ": r"\b(?:20\d{2}|19\d{2}|\d{2})[-/._](0[1-9]|1[0-2])[-/._](0[1-9]|[12]\d|3[01])T([01]\d|2[0-4])[:,.](0[0-9]|[1-5]\d)[:,.](0[0-9]|[1-5]\d)Z\b",
            "YYYY-DD-MMTHH:MM:SSZ": r"\b(?:20\d{2}|19\d{2}|\d{2})[-/._](0[1-9]|[12]\d|3[01])[-/._](0[1-9]|1[0-2])T([01]\d|2[0-4])[:,.](0[0-9]|[1-5]\d)[:,.](0[0-9]|[1-5]\d)Z\b",

            "YYYY-MM-DDTHH:MM:SS.sssZ": r"\b(?:20\d{2}|19\d{2}|\d{2})[-/._](0[1-9]|1[0-2])[-/._](0[1-9]|[12]\d|3[01])T([01]\d|2[0-4])[:,.](0[0-9]|[1-5]\d)[:,.](0[0-9]|[1-5]\d)\.(\d{3})Z\b",
            "YYYY-DD-MMTHH:MM:SS.sssZ": r"\b(?:20\d{2}|19\d{2}|\d{2})[-/._](0[1-9]|[12]\d|3[01])[-/._](0[1-9]|1[0-2])T([01]\d|2[0-4])[:,.](0[0-9]|[1-5]\d)[:,.](0[0-9]|[1-5]\d)\.(\d{3})Z\b",

            "YYYY-MM-DDTHH:MM:SS.sss±HHMM": r"\b(?:20\d{2}|19\d{2}|\d{2})[-/._](0[1-9]|1[0-2])[-/._](0[1-9]|[12]\d|3[01])T([01]\d|2[0-4])[:,.](0[0-9]|[1-5]\d)[:,.](0[0-9]|[1-5]\d)\.(\d{3})[+-](0[0-9]|1[0-2])(?::|\.|,)?(0[0-9]|[1-5]\d)\b",
            "YYYY-DD-MMTHH:MM:SS.sss±HHMM": r"\b(?:20\d{2}|19\d{2}|\d{2})[-/._](0[1-9]|[12]\d|3[01])[-/._](0[1-9]|1[0-2])T([01]\d|2[0-4])[:,.](0[0-9]|[1-5]\d)[:,.](0[0-9]|[1-5]\d)\.(\d{3})[+-](0[0-9]|1[0-2])(?::|\.|,)?(0[0-9]|[1-5]\d)\b",

            "YYYY-MM-DDTHH:MM:SS.sss±HH": r"\b(?:20\d{2}|19\d{2}|\d{2})[-/._](0[1-9]|1[0-2])[-/._](0[1-9]|[12]\d|3[01])T([01]\d|2[0-4])[:,.](0[0-9]|[1-5]\d)[:,.](0[0-9]|[1-5]\d)\.(\d{3})[+-](0[0-9]|1[0-2])\b",
            "YYYY-DD-MMTHH:MM:SS.sss±HH": r"\b(?:20\d{2}|19\d{2}|\d{2})[-/._](0[1-9]|[12]\d|3[01])[-/._](0[1-9]|1[0-2])T([01]\d|2[0-4])[:,.](0[0-9]|[1-5]\d)[:,.](0[0-9]|[1-5]\d)\.(\d{3})[+-](0[0-9]|1[0-2])\b",

            "YYYY-MM-DDTHH:MM": r"\b(?:20\d{2}|19\d{2}|\d{2})[-/._](0[1-9]|1[0-2])[-/._](0[1-9]|[12]\d|3[01])T([01]\d|2[0-4])[:,.](0[0-9]|[1-5]\d)\b",
            "YYYY-DD-MMTHH:MM": r"\b(?:20\d{2}|19\d{2}|\d{2})[-/._](0[1-9]|[12]\d|3[01])[-/._](0[1-9]|1[0-2])T([01]\d|2[0-4])[:,.](0[0-9]|[1-5]\d)\b",

            "YYYY-MM-DDTHH:MMZ": r"\b(?:20\d{2}|19\d{2}|\d{2})[-/._](0[1-9]|1[0-2])[-/._](0[1-9]|[12]\d|3[01])T([01]\d|2[0-4])[:,.](0[0-9]|[1-5]\d)Z\b",
            "YYYY-DD-MMTHH:MMZ": r"\b(?:20\d{2}|19\d{2}|\d{2})[-/._](0[1-9]|[12]\d|3[01])[-/._](0[1-9]|1[0-2])T([01]\d|2[0-4])[:,.](0[0-9]|[1-5]\d)Z\b",

            "YYYY-MM-DDTHH:MM±HHMM": r"\b(?:20\d{2}|19\d{2}|\d{2})[-/._](0[1-9]|1[0-2])[-/._](0[1-9]|[12]\d|3[01])T([01]\d|2[0-4])[:,.](0[0-9]|[1-5]\d)[+-](0[0-9]|1[0-2])(?::|\.|,)?(0[0-9]|[1-5]\d)\b",
            "YYYY-DD-MMTHH:MM±HHMM": r"\b(?:20\d{2}|19\d{2}|\d{2})[-/._](0[1-9]|[12]\d|3[01])[-/._](0[1-9]|1[0-2])T([01]\d|2[0-4])[:,.](0[0-9]|[1-5]\d)[+-](0[0-9]|1[0-2])(?::|\.|,)?(0[0-9]|[1-5]\d)\b",

            "YYYY-MM-DDTHH:MM±HH": r"\b(?:20\d{2}|19\d{2}|\d{2})[-/._](0[1-9]|1[0-2])[-/._](0[1-9]|[12]\d|3[01])T([01]\d|2[0-4])[:,.](0[0-9]|[1-5]\d)[+-](0[0-9]|1[0-2])\b",
            "YYYY-DD-MMTHH:MM±HH": r"\b(?:20\d{2}|19\d{2}|\d{2})[-/._](0[1-9]|[12]\d|3[01])[-/._](0[1-9]|1[0-2])T([01]\d|2[0-4])[:,.](0[0-9]|[1-5]\d)[+-](0[0-9]|1[0-2])\b",

            "MM-DD-YYYY HH:MM AM/PM": r"\b(?:0[1-9]|1[0-2])[-/._]?(0[1-9]|[12]\d|3[01])[-/._]?(?:20\d{2}|19\d{2}|\d{2})\s+(0[0-9]|1[0-2])[:,.]?([0-5]\d)\s*([APMapm]{2})\b",
            "DD-MM-YYYY HH:MM AM/PM": r"\b(0[1-9]|[12]\d|3[01])[-/._]?(0[1-9]|1[0-2])[-/._]?(?:20\d{2}|19\d{2}|\d{2})\s+(0[1-9]|[1][0-2])[:,.]?([0-5]\d)\s*([APMapm]{2})\b",

            "MM-DD-YYYY HH:MM": r"\b(?:0[1-9]|1[0-2])[-/._]?(0[1-9]|[12]\d|3[01])[-/._]?(?:20\d{2}|19\d{2}|\d{2})\s+([01]\d|2[0-4])[:,.]?([0-5]\d)\b",
            "DD-MM-YYYY HH:MM": r"\b(?:0[1-9]|[12]\d|3[01])[-/._]?(0[1-9]|1[0-2])[-/._]?(?:20\d{2}|19\d{2}|\d{2})\s+([01]\d|2[0-4])[:,.]?([0-5]\d)\b",

            "HH:MM:SS +/-HH:MM": r"\b(?:[01]\d|2[0-4])[:,.](?:[0-5]\d)[:,.](?:[0-5]\d)\s?([+-]\d{2}:[0-5]\d)\b",

            "HH:MM +/-HH:MM": r"\b(?:[01]\d|2[0-4])[:,.](?:[0-5]\d)\s?([+-]\d{2}:[0-5]\d)\b",

            "Day of the Week, Month Day, Year": r"\b(?:[Ss]unday|[Mm]onday|[Tt]uesday|[Ww]ednesday|[Tt]hursday|[Ff]riday|[Ss]aturday|[Ss]un|[Mm]on|[Tt]ue|[Ww]ed|[Tt]hu|[Ff]ri|[Ss]at),?\s*?(?:[Jj]anuary|[Ff]ebruary|[Mm]arch|[Aa]pril|[Mm]ay|[Jj]une|[Jj]uly|[Aa]ugust|[Ss]eptember|[Oo]ctober|[Nn]ovember|[Dd]ecember|[Jj]an|[Ff]eb|[Mm]ar|[Aa]pr|[Mm]ay|[Jj]un|[Jj]ul|[Aa]ug|[Ss]ep|[Oo]ct|[Nn]ov|[Dd]ec)\s*?\d{1,2},?\s*?\d{4}\b",

            "Day of the Week, Month Day, Year, Time": r"\b(?:[Ss]unday|[Mm]onday|[Tt]uesday|[Ww]ednesday|[Tt]hursday|[Ff]riday|[Ss]aturday|[Ss]un|[Mm]on|[Tt]ue|[Ww]ed|[Tt]hu|[Ff]ri|[Ss]at),?\s*?(?:[Jj]anuary|[Ff]ebruary|[Mm]arch|[Aa]pril|[Mm]ay|[Jj]une|[Jj]uly|[Aa]ugust|[Ss]eptember|[Oo]ctober|[Nn]ovember|[Dd]ecember|[Jj]an|[Ff]eb|[Mm]ar|[Aa]pr|[Mm]ay|[Jj]un|[Jj]ul|[Aa]ug|[Ss]ep|[Oo]ct|[Nn]ov|[Dd]ec)\s*?\d{1,2},?\s*?\d{4},\s*?\d{1,2}:\d{2}\s*([APMapm]{2})?\b",

            "Month Day, Year, Time": r"\b(?:[Jj]anuary|[Ff]ebruary|[Mm]arch|[Aa]pril|[Mm]ay|[Jj]une|[Jj]uly|[Aa]ugust|[Ss]eptember|[Oo]ctober|[Nn]ovember|[Dd]ecember|[Jj]an|[Ff]eb|[Mm]ar|[Aa]pr|[Mm]ay|[Jj]un|[Jj]ul|[Aa]ug|[Ss]ep|[Oo]ct|[Nn]ov|[Dd]ec)\s*?\d{1,2},?\s*?\d{4},\s*?\d{1,2}:\d{2}\s*([APMapm]{2})?\b",

            "HH:MM:SS.sss": r"\b([01]\d|2[0-4])[:,.](0[0-9]|[1-5]\d)[:,.](0[0-9]|[1-5]\d)\.(\d{3})\b",
            "HH:MM:SS.sss AM/PM": r"\b(?:0[0-9]|1[0-2])[:,.](?:[0-5][0-9])[:,.](?:[0-5][0-9])\.\d{3}\s*?[APap][Mm]\b",

            "HH:MM": r"\b([01]\d|2[0-4])[:,.](0[0-9]|[1-5]\d)\b",
            "HH:MM AM/PM": r"\b(?:0[0-9]|1[0-2])[:,.](?:[0-5][0-9])\s*?[APap][Mm]\b",

            "HH:MM AM/PM (Timezone)": r"^(0[0-9]|1[0-2])[:,.][0-5][0-9]( ?[APap][Mm])\s*?\([A-Za-z0-9\s:+-]+\)$",
            "HH:MM (Timezone)": r"^(?:[01]\d|2[0-4])[:,.][0-5]\d\s*?\([A-Za-z0-9\s:+-]+\)$",

            "YYYY-MM-DDTHH:MM:SS±HHMM": r"\b(?:20\d{2}|19\d{2}|\d{2})[-/._](0[1-9]|1[0-2])[-/._](0[1-9]|[12]\d|3[01])T([01]\d|2[0-4])[:,.](0[0-9]|[1-5]\d)[:,.](0[0-9]|[1-5]\d)[+-](0[0-9]|1[0-2])(?::|\.|,)?(0[0-9]|[1-5]\d)\b",
            "YYYY-DD-MMTHH:MM:SS±HHMM": r"\b(?:20\d{2}|19\d{2}|\d{2})[-/._](0[1-9]|[12]\d|3[01])[-/._](0[1-9]|1[0-2])T([01]\d|2[0-4])[:,.](0[0-9]|[1-5]\d)[:,.](0[0-9]|[1-5]\d)[+-](0[0-9]|1[0-2])(?::|\.|,)?(0[0-9]|[1-5]\d)\b",

            "YYYY-MM-DDTHH:MM AM/PM±HHMM": r"\b(?:20\d{2}|19\d{2}|\d{2})[-/._](0[1-9]|1[0-2])[-/._](0[1-9]|[12]\d|3[01])T([01]\d|2[0-4])[:,.](0[0-9]|[1-5]\d)\s*?( ?[APap][Mm])[+-](0[0-9]|1[0-2])(?::|\.|,)?(0[0-9]|[1-5]\d)\b",
            "YYYY-DD-MMTHH:MM AM/PM±HHMM": r"\b(?:20\d{2}|19\d{2}|\d{2})[-/._](0[1-9]|[12]\d|3[01])[-/._](0[1-9]|1[0-2])T([01]\d|2[0-4])[:,.](0[0-9]|[1-5]\d)\s*?( ?[APap][Mm])[+-](0[0-9]|1[0-2])(?::|\.|,)?(0[0-9]|[1-5]\d)\b"

      }

    # Initialize a counter for each date-time format group
        format_counters = {group: 0 for group in date_time_groups_new.keys()}

    # Add "other" as a separate group
        format_counters["date & time"] = 0

    # Count occurrences of each date-time format group in sampled values
        for value in sampled_values:
            matched = False
            for group, pattern in date_time_groups_new.items():
                if pd.Series([value]).str.fullmatch(pattern).any():
                    format_counters[group] += 1
                    matched = True
                    break  # Once a match is found, no need to check further

            if not matched:
                format_counters["date & time"] += 1

    # Determine the majority format group
        majority_format_group = max(format_counters, key=format_counters.get)

        return majority_format_group

############# Dimensions and Measures module ##############

    def _dim_measure_classify(self,prompt_text_DM: str, column_name: str) -> str:
        """
        - Classifies the integer or float datatype of a column to dimension, measure, or unknown.
            
        Parameters
        ----------
        prompt_text_DM (str): prompt to be passed to openai chat llm
        column_name (str): a column of the table.
            
        Returns
        -------
        final l2 model prediction

        """
        response_schemas = [self.dm_class_schema]

        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()
        prompt = ChatPromptTemplate.from_template(template=prompt_text_DM)
        messages = prompt.format_messages(column_name = column_name,
                                      format_instructions=format_instructions)
        response = self.chat_llm(messages)
        response_as_dict = output_parser.parse(response.content)
        return response_as_dict["DM_class"]

    def model_prediction(self,l1_pred: pd.DataFrame, feat_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """
        - Returns the prediction of l1 and l2 model combined.

        Parameters
        ----------
        l1_pred (pd.DataFrame): result dataframe from l1 prediction
        feat_df (pd.DataFrame): features dataframe from the dataset
        df (pd.DataFrame): dataframe obtained from the dataset

        Returns
        -------
        final prediction of both l1 and l2 combined along with all the features.
        """

        ### Merge the predicted dataframe and feature dataframe using master id
        pred_feat_df = pd.merge(feat_df, l1_pred, on = 'master_id')
    
        ######## Introducing l2: Datetype format identifier and Dimensions & Measures module ########
    
        num_samples = MODEL_CONFIG["L2PARAMS"]["SAMPLE_SIZE"]
        df_sample_size = MODEL_CONFIG["L2PARAMS"]["DF_SAMPLE_SIZE"]
    
        ### Create a dictionary of DataFrames based on unique values in "table_name" from df
        dfs = {name: pd.DataFrame({col: values for col, values in zip(group["column_name"], group["values"])})
               for name, group in df.groupby("table_name")}

        #initally setting all the l2 predictions to others
        pred_feat_df["predicted_datatype_l2"] = "others"
    
        # iterate through every row
        for index, row in pred_feat_df.iterrows():
            
            if row["predicted_datatype_l1"] == "date & time":
                col_values = df.loc[(df["dataset_name"] == row["dataset_name"]) & 
                                    (df["table_name"] == row["table_name"]) & 
                                    (df["column_name"] == row["column_name"])]["values"].iloc[0]        
                value = self.classify_datetime_format(col_values,num_samples)
                pred_feat_df.at[index,"predicted_datatype_l2"] = value
        
            elif row["predicted_datatype_l1"] == "integer" or row["predicted_datatype_l1"] == "float":
                tab_name = row["table_name"]
                col_name = row["column_name"]
                prompt_text_DM = """Class Category
                 - Dimensions contain qualitative information. These are descriptive attributes, like a product category, product key, customer address, or country. Dimensions can contain numeric characters (like an alphanumeric customer ID), but are not numeric values (It wouldn’t make sense to add up all the ID numbers in a column, for example).
                 - Measures contain quantitative values that you can measure. Measures can be aggregated.
                    Please analyze the table and classify column as either a 'Measure' or a 'Dimension' or 'Unknown'.
                INPUT
                ### TABLE: """+str(dfs[tab_name].sample(
                    min(df_sample_size,dfs[tab_name].shape[0])
                ))+"""\n\n
                ### COLUMN NAME:{column_name}\n
                ### CLASS CATEGORY:{format_instructions}"""
        
                value = self._dim_measure_classify(prompt_text_DM, column_name = col_name)
                pred_feat_df.at[index,"predicted_datatype_l2"] = row["predicted_datatype_l1"]+"_"+value.lower()
                
            else:
                pred_feat_df.at[index,"predicted_datatype_l2"] = row["predicted_datatype_l1"]

        return pred_feat_df