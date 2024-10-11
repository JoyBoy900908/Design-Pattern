import os
def GenerateUMLFig():
    print(os.getcwd())
    os.chdir("CodeToUml/jar")
    
    print(os.getcwd())
    os.system("java -jar javatouml.jar ../../custom_project original.jpg")

    os.system("java -jar javatouml.jar ../../custom_project revise.jpg")
    os.system("cd ../../")


GenerateUMLFig()