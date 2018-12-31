import re
def clean_text(text):
    # replace  . and a space with only a space, then amke all words lower case.
    text = text.replace(". "," ").replace(",","").lower()
    # get rid of the . at the end of each line. 
    cleaned_text = re.sub("\.$","",text)
    
    return cleaned_text


# extracting the number of special characters. 
def extract_special_len(text):
    # the complete special char list
    all_special_char = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    # return the special char if it is in the text
    special_list = [special_char for special_char in all_special_char\
                   if special_char in text]
    
    return len(special_list)   

# extracting the exact special characters.
def extract_special_char(text):
    # the complete special char list
    all_special_char = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    # return the special char if it is in the text
    special_list = [special_char for special_char in all_special_char\
                   if special_char in text]
    
    return special_list  


if __name__ == "__main__":
    import sys
    assert len(sys.argv)>1, "Please enter a csv file to be cleaned"
    input_file_name = sys.argv[1]
    
    # split the input file path 
    input_file_name_list = sys.argv[1].split("/")
    # combine the file path with the new file names
    output_file_name = "./Cleaned_"+input_file_name_list[-1]
    
    import pandas as pd
    print("Begin importing data for preprocessing")
    input_data = pd.read_csv(input_file_name)
    
    input_data["Cleaned_text"] = input_data["Text"].apply(clean_text)
    print("Finished text cleaning") 
    # save to disk 
    input_data.to_csv(output_file_name, index = False)
    print("Saving to disk {}".format(output_file_name))
    
    # showcase the number of special characters in the cleaned text
    input_data['Special_char_num'] = input_data["Cleaned_text"].apply(extract_special_len)
    print("Number of special characters for different PIIs after cleaning.")
    print(input_data.groupby("Labels").agg({'Special_char_num':["min","max","mean",'median']}))
    