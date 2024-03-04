import sys

def error_message_detail(error, error_detail:sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f'Error occured in python script name {file_name}, line number {exc_tb.tb_lineno}, {str(error)}'
    
    return error_message


class CustomException(Exception):
    def __init__(self, error, error_detail:sys):
        super().__init__(error)
        self.error_message = error_message_detail(error, error_detail=error_detail)
        
    def __str__(self):
        return self.error_message
    

# # Check our Custom Exception is working or not? 
# if __name__ == "__main__":
#     try:
#         a = 4/ 0
#         print(a)
#     except Exception as e:
#         raise CustomException(e, sys)
    