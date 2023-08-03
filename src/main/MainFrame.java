package main;

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.image.BufferedImage;
import java.util.Arrays;

import javax.swing.BorderFactory;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JTextArea;
import javax.swing.border.BevelBorder;

import network.CNN;

@SuppressWarnings("serial")
public class MainFrame extends JFrame
{
	private final int REAL_IMAGE_SIZE = 24;  // the real size of the image
	private final int IMAGE_STRETCH = 12;
	private final int IMAGE_SIZE = REAL_IMAGE_SIZE * IMAGE_STRETCH;  // the size of the image displayed on screen
	private final Font BUTTON_FONT = new Font("Serif", Font.BOLD, 18);
	private final Font TEXT_FONT = new Font("Serif", Font.PLAIN, 16);
	private BufferedImage image;  // the image of the drawn digit
	private JLabel imageLabel;  // the label carrying the stretched image
	private JTextArea outputText;  // tells you which digit it is
	private CNN cnn;
	
	public MainFrame(CNN cnn)
	{
		this.cnn = cnn;
		
		setSize(520, 420);
        setResizable(false);
        setLayout(null);
        setTitle("Digit Recognizer");
        setLocationRelativeTo(null);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setVisible(true);
        
        image = new BufferedImage(IMAGE_SIZE, IMAGE_SIZE, BufferedImage.TYPE_INT_RGB);
        
        // initialize gui components
        imageLabel = new JLabel();
        add(imageLabel);
        imageLabel.setBounds(40, 40, IMAGE_SIZE, IMAGE_SIZE);
        imageLabel.setBorder(BorderFactory.createBevelBorder(BevelBorder.LOWERED));
        imageLabel.setOpaque(true);
        imageLabel.addMouseListener(new MouseListener()
        {
			@Override
			public void mouseClicked(MouseEvent e)
			{
				drawImage(e.getX(), e.getY());
			}

			@Override
			public void mousePressed(MouseEvent e) {}
			@Override
			public void mouseReleased(MouseEvent e) {}
			@Override
			public void mouseEntered(MouseEvent e) {}
			@Override
			public void mouseExited(MouseEvent e) {}
        });
        imageLabel.addMouseMotionListener(new MouseMotionListener()
        {
			@Override
			public void mouseDragged(MouseEvent e)
			{
				drawImage(e.getX(), e.getY());
			}
			
			@Override
			public void mouseMoved(MouseEvent e) {}
        });
        imageLabel.setVisible(true);
        
        JButton clearButton = new JButton("Clear");
        add(clearButton);
        clearButton.setFont(BUTTON_FONT);
        clearButton.setBounds(360, 30, 120, 30);
        clearButton.addActionListener(new ActionListener()
        {
			@Override
			public void actionPerformed(ActionEvent e)
			{
				resetImage();
			}
        });
        clearButton.setVisible(true);
        
        JButton predButton = new JButton("Predict");
        add(predButton);
        predButton.setFont(BUTTON_FONT);
        predButton.setBounds(360, 70, 120, 30);
        predButton.addActionListener(new ActionListener()
        {
			@Override
			public void actionPerformed(ActionEvent e)
			{
				// fetch image
				double[][][] x = new double[1][REAL_IMAGE_SIZE][REAL_IMAGE_SIZE];
				for(int i = 0; i < REAL_IMAGE_SIZE; i++)
				{
					for(int j = 0; j < REAL_IMAGE_SIZE; j++)
					{
						int clr = image.getRGB(i * IMAGE_STRETCH, j * IMAGE_STRETCH);
						int rgb = clr & 0x000000ff;
						x[0][j][i] = (255.0 - rgb) / 255;  // swap i & j
					}
				}
				// pad x, this step can promote accuracy
				double[][][] paddedX = new double[1][28][28];
				int pad = 28 - REAL_IMAGE_SIZE;
				for(int h = 0; h < REAL_IMAGE_SIZE; h++)
				{
					for(int w = 0; w < REAL_IMAGE_SIZE; w++)
					{
						paddedX[0][pad + h][pad + w] = x[0][h][w];
					}
				}
				
				// get prediction from cnn
				double[] prediction = cnn.predict(paddedX);
				System.out.println("Prediction:\r\n" + Arrays.toString(prediction));
				// determining the predicted digit
				outputText.setText("");
				int maxIndex = 0;
				double maxValue = 0;
				for(int i = 0; i < prediction.length; i++)
				{
					if(prediction[i] > maxValue)
					{
						maxValue = prediction[i];
						maxIndex = i;
					}
				}
				outputText.append("Prediction: " + maxIndex + "\r\n");
				
				// displaying probability for each digit
				for(int i = 0; i < prediction.length; i++)
				{
					int numChars = (int) (10 - 10 * Math.log10(prediction[i]) / -5);
					outputText.append(i + ": ");
					for(int j = 0; j < numChars; j++)
					{
						outputText.append("^");
					}
					outputText.append("\r\n");
				}
			}
        });
        predButton.setVisible(true);
        
        outputText = new JTextArea();
        add(outputText);
        outputText.setFont(TEXT_FONT);
        outputText.setLineWrap(true);
        outputText.setBounds(370, 120, 110, 220);
        outputText.setOpaque(false);
        outputText.setEditable(false);
        outputText.setVisible(true);
        outputText.setText("Prediction: 8\r\n0: ^^^^^^^^^^\r\n0: ^^^^^^^^^^\r\n0: ^^^^^^^^^^\r\n0: ^^^^^^^^^^\r\n0: ^^^^^^^^^^\r\n0: ^^^^^^^^^^\r\n0: ^^^^^^^^^^\r\n0: ^^^^^^^^^^\r\n0: ^^^^^^^^^^\r\n9: ^^^^^^^^^^");
        
        // set image to all white
        resetImage();
	}
	
	// draw a pixel on image with black paint
	private void drawImage(int x, int y)
	{
		Graphics2D g2d = image.createGraphics();
        g2d.setPaint(Color.black);
        int realX = x / IMAGE_STRETCH;
        int realY = y / IMAGE_STRETCH;
        g2d.fillRect(realX * IMAGE_STRETCH, realY * IMAGE_STRETCH, IMAGE_STRETCH, IMAGE_STRETCH);
        // implementing a bigger brush effect
        g2d.fillRect(Math.min((realX + 1), REAL_IMAGE_SIZE) * IMAGE_STRETCH, realY * IMAGE_STRETCH, IMAGE_STRETCH, IMAGE_STRETCH);
        g2d.fillRect(Math.max((realX - 1), 0) * IMAGE_STRETCH, realY * IMAGE_STRETCH, IMAGE_STRETCH, IMAGE_STRETCH);
        g2d.fillRect(realX * IMAGE_STRETCH, Math.min((realY + 1), REAL_IMAGE_SIZE) * IMAGE_STRETCH, IMAGE_STRETCH, IMAGE_STRETCH);
        g2d.fillRect(realX * IMAGE_STRETCH, Math.max((realY - 1), 0) * IMAGE_STRETCH, IMAGE_STRETCH, IMAGE_STRETCH);
        
        imageLabel.setIcon(new ImageIcon(image));
	}
	
	// set the image to be all white
	private void resetImage()
	{
		Graphics2D g2d = image.createGraphics();
        g2d.setPaint(Color.white);
        g2d.fillRect(0, 0, IMAGE_SIZE, IMAGE_SIZE);
        g2d.dispose();
        
        imageLabel.setIcon(new ImageIcon(image));
        outputText.setText("");
	}
}
