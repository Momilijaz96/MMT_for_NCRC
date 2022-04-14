import matplotlib.pyplot as plt

def get_plot(PATH,A,B,title,lblA,lblB,xlabel='Epochs',ylabel='Mean Loss',savefig=True,showfig=False):
	plt.figure()
	plt.plot(A,'-bo',label=lblA)
	plt.plot(B,'-ro',label=lblB)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.legend()
	plt.savefig(PATH+title+'.png')
