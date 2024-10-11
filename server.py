import asyncio
import openai
import os
import json
from websockets.server import serve
import predict

IPADDR="localhost"
PORT=8080
KEY_PATH="apikey.txt"
MODEL="gpt-4-turbo"
message=[
        {"role":"system","content":"你是一個可以點出design pattern瑕疵的強力助手。"}
        ]

class UML:
    def GenerateUML(self):
        pass

    def SetOriginalCode(self,code):
        self.originalCode=code

    def SetReviceCode(self,code):
        self.reviceCode=code

    def GenerateUMLFig(self):
        print("----------------------------------------generateoriuml----------------------------------------")
        os.chdir("CodeToUml/jar")
        os.system("java -jar javatouml.jar ../../custom_project original")
        print("----------------------------------------generaterevuml----------------------------------------")
        with open("../../custom_project/test.java","w") as f:
                f.write(self.reviceCode)
        os.system("java -jar javatouml.jar ../../custom_project revise")
        os.chdir("../../")

class GPT:

    def __init__(self,key,model):
        openai.api_key=key
        self.model=model
    
    def PredictSuggestion(self,originalCode,pattern):
        
        content=f"基於我給的程式碼幫我點出 {pattern} pattern的瑕疵並給我完整的程式碼"
        message.append({"role":"user","content":originalCode})
        message.append( {"role":"user","content":content})
        completion=openai.ChatCompletion.create(
            model=self.model,
            messages=message
            )
        # print(completion['choices'][0]["message"]["content"])  
        response=completion['choices'][0]["message"]["content"]
        message.append( {"role":"assistant","content":response})
        return response
    
    def Prediciton(self,userPrompt):
        message.append({"role":"user","content":userPrompt})
        completion=openai.ChatCompletion.create(
            model=self.model,
            messages=message
            )
        response=completion['choices'][0]["message"]["content"]
        message.append( {"role":"assistant","content":response})
        return response

class DataPreprocess:

    def __init__(self,key_path):
        self.key_path=key_path
    
    def initialize(self):
        self.key=self.ReadKey(self.key_path)

    def ReadKey(self,path):
        f=open(path,'r')
        key = f.readline()
        f.close()
        return key

    def GetResponseCode(self,response):
        print("----------------------------------------GetResponseCode running----------------------------------------")
        allResponse=response.split("\n")
        startIndex=0
        endIndex=0
        for i in range(len(allResponse)):
            if "```java" in allResponse[i]:
                startIndex=i+1
            elif "```" in allResponse[i]:
                endIndex=i
        print(allResponse[startIndex:endIndex])
        print(allResponse[0:startIndex-1]+allResponse[endIndex+1:])
        return allResponse[startIndex:endIndex],allResponse[0:startIndex-1]+allResponse[endIndex+1:]

    def MergeArray(self,arr):
        buffer=""
        for i in iter(arr):
            buffer+=i+"\n"
        return buffer

class Content:
    def __init__(self):
        self.allData=DataPreprocess(KEY_PATH)
        self.allData.initialize()
        self.gpt=GPT(self.allData.key,MODEL)
        self.webSocket=WebSocket(self)
        self.uml=UML()

    def Flow(self):
        asyncio.run(self.webSocket.SocketInit())

class WebSocket:

    def __init__(self,content):
        self.content=content

    # async def flow(self,socket,data):
    #     if data["flag"] == "predict":
    #         self.content.uml.SetOriginalCode(data["originalCode"])
    #         with open("./custom_project/test.java","w") as f:
    #             f.write(data["originalCode"])
    #             pattern=""
    #         try:
    #             pattern=predict.run_custom_project_predict()
    #             print(pattern)
    #             await self.SendStatus(socket,str(pattern))
    #         except:
    #             pattern="the wrong java code"
    #             await self.SendStatus(socket,str(pattern))
    #         await self.SendStatus(socket,self.content.gpt.Prediction(data["originalCode"],pattern))
    #         # await socket.send(pattern)
    #         # await socket.send(self.content.gpt.Prediction(data["originalCode"],pattern))
    #     elif data["flag"]== "generateUML":
    #         self.content.uml.SetReviceCode(data["code"])
    #         self.content.uml.GenerateUMLFig()
    #         await self.SendStatus(socket,"done uml")
    
    async def flow(self,socket,code):
        try:
            pattern=self.PredictPattern(code)
            print("PredictPattern END")
            response=self.content.gpt.PredictSuggestion(code,pattern)
            generatedCode,responseMessage=self.content.allData.GetResponseCode(response)
            self.content.uml.SetReviceCode(self.content.allData.MergeArray(generatedCode))
            self.content.uml.GenerateUMLFig()
            await self.SendStatus(socket,json.dumps([self.content.allData.MergeArray(generatedCode),self.content.allData.MergeArray(responseMessage),pattern]))

        except:
            print("except")
            responses = self.content.gpt.Prediciton(code)
            try:
                generatedCode,responseMessage = self.content.allData.GetResponseCode(responses)
                self.content.uml.SetReviceCode(self.content.allData.MergeArray(generatedCode))
                self.content.uml.GenerateUMLFig()
            except:
                await self.SendStatus(socket,json.dumps(["None",responses]))
            await self.SendStatus(socket,json.dumps(["None",responses]))

    def PredictPattern(self,code):
        self.content.uml.SetOriginalCode(code)
        with open("./custom_project/test.java","w") as f:
            f.write(code)
        try:
            pattern=predict.run_custom_project_predict()
            return pattern
        except:
            raise ("error")

    def GenerateUml(self,code):
        self.content.uml.SetReviceCode(code)
        self.content.uml.GenerateUMLFig()
           
    async def AcceptConnection(self,socket):
        async for message in socket:
            #print(message)
            data=json.loads(message)
            await self.flow(socket,data)

    async def SendStatus(self,socket,status):
        await socket.send(status)

    async def SocketInit(self):
        async with serve(self.AcceptConnection, IPADDR, PORT):
            await asyncio.Future() 
      
def main():
    content=Content()
    print(f"server start.Listening on {IPADDR}:{PORT}")
    content.Flow()
    
if __name__=="__main__":
    main()