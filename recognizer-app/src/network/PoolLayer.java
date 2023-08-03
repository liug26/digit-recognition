package network;

public class PoolLayer implements Layer
{
	private int numChannels;
	private int xHeight;
	private int xWidth;
	private int fHeight;
	private int fWidth;
	private int stride;
	private int yHeight;
	private int yWidth;
	
	public PoolLayer(int[] xShape, int[] filterShape, int stride, String poolType)
	{
		// storing variables
		this.numChannels = xShape[0];
		this.xHeight = xShape[1];
		this.xWidth = xShape[2];
		this.fHeight = filterShape[0];
		this.fWidth = filterShape[1];
		this.stride = stride;
		// dimensions of output matrix
		this.yHeight = (this.xHeight - this.fHeight) / this.stride + 1;
		this.yWidth = (this.xWidth - this.fWidth) / this.stride + 1;
	}
	
	@Override
	public Object forward(Object objX)
	{
		// assuming x's shape matches xShape
		double[][][] x = (double[][][]) objX;
		double[][][] y = new double[this.numChannels][this.yHeight][this.yWidth];
		
		for(int c = 0; c < this.numChannels; c++)
		{
			for(int yH = 0; yH < this.yHeight; yH++)
			{
				for(int yW = 0; yW < this.yWidth; yW++)
				{
					double a = 0;
					for(int i = 0; i < this.fHeight; i++)
					{
						for(int j = 0; j < this.fWidth; j++)
						{
							// only average pooling implemented
							double aSlice = x[c][yH * this.stride + i][yW * this.stride + j];
							a += 1.0 / this.fHeight / this.fWidth * aSlice;
						}
					}
					y[c][yH][yW] = a;
				}
			}
		}
		
		return y;
	}

	@Override
	public int paramsCount()
	{
		return 0;
	}

	@Override
	public void load(Double[] params) {}
}
