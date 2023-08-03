package network;

public interface Layer
{
	public Object forward(Object objX);
	public int paramsCount();
	public void load(Double[] params);
}
