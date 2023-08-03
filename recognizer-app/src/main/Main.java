package main;

import network.CNN;

public class Main
{
	private static CNN cnn;
	private static MainFrame mainFrame;
	
	public static void main(String args[])
	{
		cnn = new CNN();
		NetworkIO.load(cnn);
		mainFrame = new MainFrame(cnn);
	}
}
