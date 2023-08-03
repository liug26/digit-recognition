package network;

public class ConvLayer implements Layer
{
	private int numChannels;
	private int xHeight;
	private int xWidth;
	private int pad;
	private int fHeight;
	private int fWidth;
	private int numFilters;
	private int stride;
	private String activationFunc;
	private double[][][][] filters;
	private double[] b;
	private int yHeight;
	private int yWidth;
	
	public ConvLayer(int[] xShape, int pad, int[] filterShape, int numFilters, int stride, String activationFunc)
	{
		// storing variables
		this.numChannels = xShape[0];
		this.xHeight = xShape[1];
		this.xWidth = xShape[2];
		this.pad = pad;
		this.fHeight = filterShape[0];
		this.fWidth = filterShape[1];
		this.numFilters = numFilters;
		this.stride = stride;
		this.activationFunc = activationFunc;
		
		// initialize weights
		filters = new double[this.numFilters][this.numChannels][this.fHeight][this.fWidth];
		b = new double[this.numFilters];
		// dimensions of output matrix
		yHeight = (this.xHeight - this.fHeight + 2 * this.pad) / this.stride + 1;
		yWidth = (this.xWidth - this.fWidth + 2 * this.pad) / this.stride + 1;
	}
	
	@Override
	public Object forward(Object objX)
	{
		// assuming x's shape matches xShape
		double[][][] x = (double[][][]) objX;
		double[][][] y = new double[this.numFilters][this.yHeight][this.yWidth];
		
		if(this.pad != 0)
		{
			x = pad(x);
		}
		
		for(int f = 0; f < this.numFilters; f++)
		{
			for(int yH = 0; yH < this.yHeight; yH++)
			{
				for(int yW = 0; yW < this.yWidth; yW++)
				{
					double z = 0;
					// do convolution
					for(int c = 0; c < this.numChannels; c++)
					{
						for(int i = 0; i < this.fHeight; i++)
						{
							for(int j = 0; j < this.fWidth; j++)
							{
								double aSlice = x[c][yH * this.stride + i][yW * this.stride + j];
								z += aSlice * this.filters[f][c][i][j];
							}
						}
					}
					z += this.b[f];
					y[f][yH][yW] = CNN.activation(z, this.activationFunc);
				}
			}
		}
		
		return y;
	}
	
	private double[][][] pad(double[][][] x)
	{
		double[][][] paddedX = new double[this.numChannels][this.xHeight + this.pad * 2][this.xWidth + this.pad * 2];
		for(int c = 0; c < this.numChannels; c++)
		{
			for(int h = 0; h < this.xHeight; h++)
			{
				for(int w = 0; w < this.xWidth; w++)
				{
					paddedX[c][this.pad + h][this.pad + w] = x[c][h][w];
				}
			}
		}
		return paddedX;
	}

	@Override
	public int paramsCount()
	{
		return this.numFilters * this.numChannels * this.fHeight * this.fWidth + this.numFilters;
	}

	@Override
	public void load(Double[] params)
	{
		int index = 0;
		
		for(int f = 0; f < this.numFilters; f++)
		{
			for(int c = 0; c < this.numChannels; c++)
			{
				for(int h = 0; h < this.fHeight; h++)
				{
					for(int w = 0; w < this.fWidth; w++)
					{
						this.filters[f][c][h][w] = params[index++];
					}
				}
			}
		}
		
		for(int f = 0; f < this.numFilters; f++)
		{
			this.b[f] = params[index++];
		}
		
		if(index != params.length) System.out.println("not all params loaded in conv layer");
	}
}
