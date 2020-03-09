#READ collum + row in excel to n-D list in 2 ways
import xlrd
import torch

workbook = xlrd.open_workbook("Data.xlsx")
sheet = workbook.sheet_by_name("Sheet1")

rowcount = sheet.nrows
colcount = sheet.ncols

# print(rowcount)
# print(colcount)

Data =[]
Output = []
Data_Compress = []

for curr_col in range(0, colcount, 1): #5300 columns
    col_data = []
    col_data_out =[]

    for curr_row in range(0, rowcount, 1): #3 rows
        if curr_row != 2:
            data = sheet.cell_value(curr_row, curr_col) #take data from 1 single cell
            col_data.append(data)
        else:
            data_out = sheet.cell_value(curr_row, curr_col) #take data from 1 single cell
            col_data_out.append(data_out)
            
    Data.append(col_data)
    Data.append(col_data_out)


# print(Data)

for i in range(0,len(Data),2):
    data_select = []
    for j in range(2):
        data_select.append(Data[i+j])
    
    Data_Compress.append(data_select)

    
# print(Output)
# print(len(Data))

# Data will store like this Data = [[0.737784474287975, 0.288032444846822], [1.0]], [[1.24002004818493, 0.0990697069733273], [1.0]]]


