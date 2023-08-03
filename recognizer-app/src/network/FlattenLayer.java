package network;

public class FlattenLayer implements Layer
{
	private int numChannels;
	private int xHeight;
	private int xWidth;
	
	public FlattenLayer(int[] xShape)
	{
		// storing variables
		this.numChannels = xShape[0];
		this.xHeight = xShape[1];
		this.xWidth = xShape[2];
	}

	@Override
	public Object forward(Object objX)
	{
		// assuming x's shape matches xShape
		double[][][] x = (double[][][]) objX;
		double[] y = new double[this.numChannels * this.xHeight * this.xWidth];
		int index = 0;
		
		/* if x is formatted (#channels, height, width)
		for(int i = 0; i < x.length; i++)
		{
			for(int j = 0; j < x[i].length; j++)
			{
				for(int k = 0; k < x[i][j].length; k++)
				{
					y[index] = x[i][j][k];
					index++;
				}
			}
		}
		*/
		
		for(int j = 0; j < x[0].length; j++)
		{
			for(int k = 0; k < x[0][j].length; k++)
			{
				for(int i = 0; i < x.length; i++)
				{
					y[index] = x[i][j][k];
					index++;
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
