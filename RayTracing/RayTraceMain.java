import RayTracingPackages.*;

//CREATE THE OPTION TO ADD BOUNDARIES FROM CENTER FOR PLANES
//CONSIDER ADDING OPTIONS FOR MORE OBJECTS
//CONSIDER ADDING PATTERNS FOR PLANES AND TILING
//CHECK PLANAR ROTATIONS (SEEMS NOT WORKING)

public class RayTraceMain {
    public static void main (String args[]) {
        System.out.println("---------------------");
        System.out.println("VECTOR TEST:");
        Vector testVector = new Vector(5,7,9,0,0,0);
        testVector.normalize();

        double cosX = Math.cos(Math.toRadians(50));
        double cosY = Math.cos(Math.toRadians(0));
        double sinX = Math.sin(Math.toRadians(50));
        double sinY = Math.sin(Math.toRadians(0));

        testVector.rotate(cosX,cosY,sinX,sinY);
        System.out.println(testVector);

        System.out.println("---------------------");

        boolean running = true;
        World testWorld = new World("sky blue");
        Camera testCam = new Camera(testWorld,800,800,0,0,0,90,-20,-5);
        Display testDisplay = new Display(testCam);
        Ellipse testEllipse = new Ellipse(testWorld,0,0,5,1,1,1,1,"blue");
        Plane testPlane = new Plane(testWorld,0,-10,0,5,5,5,0,0,0,210,10,10);
        Plane testPlane2 = new Plane(testWorld,2.5,-7.5,0,5,5,5,90,90,0,210,10,10);
        Plane testPlane3 = new Plane(testWorld,-2.5,-7.5,0,5,5,5,90,90,0,210,10,10);
        Plane testPlane4 = new Plane(testWorld,0,-5,0,5,5,5,0,0,0,210,10,10);

        Point testPoint = new Point(2,5,7);
        System.out.println(testPoint);
        long startTime = System.nanoTime();
        int frame = 0;
        long frameRateTimerStart = 0;
        long frameRateTimerEnd = 0;
        while(running) {
            while(!testDisplay.isPaused()){
                if(frame==0){
                    frameRateTimerStart = System.nanoTime();
                    frame+=1;
                } else if (frame==60) {
                    frameRateTimerEnd = System.nanoTime();
                    System.out.println("TIME TO RENDER 60 FRAMES: " + ((double)(frameRateTimerEnd-frameRateTimerStart)/1000000000));
                    frame = 0;
                } else {
                    frame+=1;
                }
                testCam.renderCameraView();
                testDisplay.updateImage();

                try {
                    Thread.sleep(0);
                } catch (InterruptedException e){
                    System.out.println("ERROR");
                }

            }

            try {
            Thread.sleep(100);
            }  catch (InterruptedException e){
            System.out.println("ERROR");
            }
        
        }
        
        long endTime = System.nanoTime();
        System.out.println(endTime-startTime);
    }
}