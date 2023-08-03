package network;

public class DenseLayer implements Layer
{
	private int numNeurons;
	private int numInputs;
	private String activationFunc;
	private double[][] w;
	private double[] b;
	
	public DenseLayer(int numNeurons, int numInputs, String activationFunc)
	{
		// store variables
		this.numNeurons = numNeurons;
		this.numInputs = numInputs;
		this.activationFunc = activationFunc;
		
		//
		this.w = new double[this.numNeurons][this.numInputs];
		this.b = new double[this.numNeurons];
	}
	
	@Override
	public Object forward(Object objX)
	{
		// assuming x's shape matches xShape
		double[] x = (double[]) objX;
		double[] y = new double[this.numNeurons];
		double sumE = 0;
		
		for(int n = 0; n < this.numNeurons; n++)
		{
			double z = 0;
			for(int i = 0; i < this.numInputs; i++)
			{
				z += x[i] * w[n][i];
			}
			z += b[n];
			
			y[n] = CNN.activation(z, this.activationFunc);
			if(this.activationFunc.equals("softmax")) sumE += Math.exp(z);
		}
		
		if(this.activationFunc.equals("softmax"))
		{
			for(int n = 0; n < this.numNeurons; n++)
			{
				y[n] = Math.exp(y[n]) / sumE;
			}
		}
		
		return y;
	}

	@Override
	public int paramsCount()
	{
		return this.numNeurons * this.numInputs + this.numNeurons;
	}

	@Override
	public void load(Double[] params)
	{
		int index = 0;
		
		for(int n = 0; n < this.numNeurons; n++)
		{
			for(int i = 0; i < this.numInputs; i++)
			{
				this.w[n][i] = params[index++];
			}
		}
		
		for(int n = 0; n < this.numNeurons; n++)
		{
			this.b[n] = params[index++];
		}
		
		if(index != params.length) System.out.println("not all params loaded in dense layer");
	}
}
