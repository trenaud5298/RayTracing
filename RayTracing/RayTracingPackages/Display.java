package RayTracingPackages;

import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.image.BufferedImage;
import javax.swing.*;
import java.util.Random;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionListener;
import java.awt.Cursor;
import java.awt.Toolkit;
import java.awt.AWTException;
import java.awt.Robot;
import java.awt.event.InputEvent;

public class Display {
    private Camera viewCamera;
    private int width;
    private int height;
    private int halfWidth;
    private int halfHeight;
    private double fovScale;
    private JFrame frame;
    private ImagePanel imagePanel;
    private boolean movingForward = false;
    private boolean movingBackward = false;
    private boolean movingRight = false;
    private boolean movingLeft = false;
    private boolean movingUp = false;
    private boolean movingDown = false;
    private boolean paused = false;
    private int mousePosX = 0;
    private int mousePosY = 0;
    private int mouseSensitivity = 30;
    private static Cursor transparentCursor = Toolkit.getDefaultToolkit().createCustomCursor(Toolkit.getDefaultToolkit().getImage(""),new java.awt.Point(0, 0),"InvisibleCursor");
    
    public Display (Camera viewCamera) {
        this.viewCamera = viewCamera;
        this.createFrame();
        this.width = viewCamera.getWidth();
        this.height = viewCamera.getHeight();
        this.halfWidth = (int)this.width/2;
        this.halfHeight = (int)this.height/2;
    }

    private void createFrame() {
        // Display the image in a JFrame
        frame = new JFrame("Image Display");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        imagePanel = new ImagePanel();
        frame.add(imagePanel);
        frame.setSize(viewCamera.getWidth(), viewCamera.getHeight());
        frame.setVisible(true);
        frame.addKeyListener(new MyKeyListener());
        frame.setFocusable(true);
        frame.requestFocusInWindow();
        frame.addMouseMotionListener(new MyMouseMotionListener(this));
        frame.setCursor(transparentCursor);
    }


    public void updateImage() {
        //MOVEMENT HANDLING
        if(movingForward) {
            this.viewCamera.moveForward();
        }
        if(movingBackward) {
            this.viewCamera.moveBackward();
        }
        if(movingRight) {
            this.viewCamera.moveRight();
        }
        if(movingLeft) {
            this.viewCamera.moveLeft();
        }
        if(movingUp) {
            this.viewCamera.moveUp();
        }
        if(movingDown) {
            this.viewCamera.moveDown();
        }
        //LOOKING HANDLING
        double mouseChangeX = this.mousePosX-this.halfWidth;
        double mouseChangeY = this.mousePosY-this.halfHeight;
        double horizontalRotationAngle = (double) mouseChangeX*this.mouseSensitivity/this.halfWidth;
        double verticalRotationAngle = (double) mouseChangeY*this.mouseSensitivity/this.halfHeight;
        this.viewCamera.turnHorizontally(horizontalRotationAngle);
        this.viewCamera.turnVertically(verticalRotationAngle);
        try {
            // Create a Robot object
            Robot robot = new Robot();
            robot.mouseMove(this.halfWidth, this.halfHeight);

        } catch (AWTException e) {
            e.printStackTrace();
        }
        //UPDATE IMAGE DISPLAY
        imagePanel.setImage(viewCamera.getImage());
    }

    public boolean isPaused() {
        return paused;
    }


    private class ImagePanel extends JPanel {
        private BufferedImage img;

        public void setImage (BufferedImage img) {
            this.img = img;
            repaint();
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            g.drawImage(img, 0, 0, this);
        }
    }

    private class MyKeyListener implements KeyListener {
        //KEY PRESS
        @Override
        public void keyPressed(KeyEvent e) {
            if(e.getKeyCode() == KeyEvent.VK_W) {
                movingForward = true;
            }
            else if(e.getKeyCode() == KeyEvent.VK_A) {
                movingLeft = true;
            }
            else if(e.getKeyCode() == KeyEvent.VK_S) {
                movingBackward = true;
            }
            else if(e.getKeyCode() == KeyEvent.VK_D) {
                movingRight = true;
            }
            else if(e.getKeyCode() == KeyEvent.VK_SPACE) {
                movingUp = true;
            }
            else if(e.getKeyCode() == KeyEvent.VK_SHIFT) {
                movingDown = true;
            }
            else if(e.getKeyCode() == KeyEvent.VK_ESCAPE) {
                if(paused){
                    paused = false;
                    frame.setCursor(transparentCursor);
                } else {
                    paused = true;
                    frame.setCursor(Cursor.getDefaultCursor());
                }
            }
        }

        //KEY RELEASE
        @Override
        public void keyReleased(KeyEvent e) {
            if(e.getKeyCode() == KeyEvent.VK_W) {
                movingForward = false;
            }
            else if(e.getKeyCode() == KeyEvent.VK_A) {
                movingLeft = false;
            }
            else if(e.getKeyCode() == KeyEvent.VK_S) {
                movingBackward = false;
            }
            else if(e.getKeyCode() == KeyEvent.VK_D) {
                movingRight = false;
            }
            else if(e.getKeyCode() == KeyEvent.VK_SPACE) {
                movingUp = false;
            }
            else if(e.getKeyCode() == KeyEvent.VK_SHIFT) {
                movingDown = false;
            }
        }

        //KEY TYPE (NOTE:NOT IMPLIMENTED YET BUT MAY BE USED WHEN MAKING UI)
        @Override
        public void keyTyped(KeyEvent e) {
            if(e.getKeyCode() == KeyEvent.VK_W) {
                
            }
        }
    }

    private static class MyMouseMotionListener implements MouseMotionListener {
        private Display display;

        public MyMouseMotionListener(Display display){
            this.display = display;
        }
        @Override
        public void mouseDragged(MouseEvent e) {
            // This method is called when the mouse is dragged (moved with a button held down)
            display.mousePosX = e.getX();
            display.mousePosY = e.getY();
        }

        @Override
        public void mouseMoved(MouseEvent e) {
            // This method is called when the mouse is moved (without any buttons held down)
            display.mousePosX = e.getX();
            display.mousePosY = e.getY();
        }
    }

}