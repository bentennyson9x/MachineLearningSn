import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from mpl_toolkits.mplot3d import axes3d
class Expression :
    stack = [];
    def __init__(self, polish):
        self.polish = polish


    def isOperator(seft, who):
        if (who == "+" or who == "-" or who == "*"
                or who == "/" or who == "^" or who =="(" or who==")"):
            return 1
        return 0
    def getPiority(self, char):
        if (char=="*" or char=="/" or char=="%"):
            return 2
        elif (char =="^") :
            return 3
        elif (char=="+" or char =="-") :
            return 1
        elif (char=="("):
            return 0
        return 0

    def Prefix2PostFix(self):
        char="";
        strTemp="";
        result="";
        for i in range (len(self.polish)) :
            char = self.polish[i]
            if (self.isOperator(char)!=1 or char.isalpha() ) :
                strTemp+=char
            else:
                result+=strTemp
                strTemp=""
                if (char=="(") :
                    self.stack.append(char)
                else:
                    if (char==")") :
                        while (self.stack[-1]!="("):
                            result+=self.stack.pop();
                        if (self.stack[-1]=="(") :self.stack.pop()
                    else:
                        while(len(self.stack)!=0 and self.getPiority(char)<=self.getPiority(self.stack[-1])):
                            result+=self.stack.pop()
                        self.stack.append(char);
        if(strTemp!=""): result+=strTemp
        while(len(self.stack)!=0):
            char=self.stack.pop()
            if (char!="("):
                result+=char
        print(result)
        return 0