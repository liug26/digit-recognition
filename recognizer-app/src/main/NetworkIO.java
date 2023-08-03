package main;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;

import network.CNN;

public class NetworkIO
{
	private static final String LOAD_PATH = System.getProperty("user.dir") + "/network.txt";
	
	public static void load(CNN cnn)
	{
		File loadFile = new File(LOAD_PATH);
	    Scanner reader;
	    ArrayList<Double> params = new ArrayList<Double>();
		try
		{
			reader = new Scanner(loadFile);
			while (reader.hasNextLine())
		    {
		    	String line = reader.nextLine();
		        if(!line.equals(""))
		        {
		        	String[] splitedLine = line.split(", ");
		        	for(String str : splitedLine)
		        	{
		        		params.add(Double.parseDouble(str));
		        	}
		        }
		    }
			reader.close();
			
		}
		catch (FileNotFoundException e)
		{
			e.printStackTrace();
		}
		
		cnn.load(params);
	}
}
