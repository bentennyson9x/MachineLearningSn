# a,c,d,e,x,z=0,0,0,0,0,0
# a = input("Please enter number a : ");
# c = input ("Please enter number c : ");
# d = input("Please enter number d : ");
# e = input ("Please enter number e : ");
# x = input("Please enter number x : ");
# z = input ("Please enter number z : ");
# stack = [];
# polish = ""
from trungtohauto.Expression import Expression


def compile():
    expression =  Expression("a*(B+C-D/E)/F#")
    expression.Prefix2PostFix()

    return
compile()