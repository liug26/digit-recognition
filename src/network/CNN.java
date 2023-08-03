package network;

import java.util.ArrayList;
import java.util.List;

public class CNN
{
	private Layer[] network;
	
	public CNN()
	{
		// a slightly modified LeNet structure
		network = new Layer[8];
		network[0] = new ConvLayer(new int[] {1, 28, 28}, 2, new int[] {5, 5}, 6, 1, "relu");
		network[1] = new PoolLayer(new int[] {6, 28, 28}, new int[] {2, 2}, 2, "average");
		network[2] = new ConvLayer(new int[] {6, 14, 14}, 0, new int[] {5, 5}, 16, 1, "relu");
		network[3] = new PoolLayer(new int[] {16, 10, 10}, new int[] {2, 2}, 2, "average");
		network[4] = new FlattenLayer(new int[] {16, 5, 5});
		network[5] = new DenseLayer(120, 400, "relu");
		network[6] = new DenseLayer(84, 120, "relu");
		network[7] = new DenseLayer(10, 84, "softmax");
	}
	
	public double[] predict(double[][][] x)
	{
		Object a = x;
		for(Layer layer : network)
		{
			a = layer.forward(a);
		}
		return (double[]) a;
	}
	
	public void load(ArrayList<Double> data)
	{
		int index = 0;
		
		for(Layer layer : network)
		{
			int paramsCount = layer.paramsCount();
			List<Double> params = data.subList(index, index + paramsCount);
			layer.load(params.toArray(new Double[0]));
			index += paramsCount;
		}
		
		if(index != data.size()) System.out.println("not all data loaded");
		System.out.println("network loaded");
	}
	
	public static double activation(double z, String activationFunc)
	{
		if(activationFunc.equals("relu"))
		{
			return Math.max(z, 0);
		}
		else if(activationFunc.equals("sigmoid"))
		{
			return 1 / (1 + Math.exp(-z));
		}
		else if(activationFunc.equals("softmax"))
		{
			// this requires a more complex implementation
			return z;
		}
		else
		{
			System.out.println("unrecognized activation function: " + activationFunc);
			return 0;
		}
	}
}
