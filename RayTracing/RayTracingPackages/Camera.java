package RayTracingPackages;

import java.awt.image.BufferedImage;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import RayTracingPackages.Vector;

public class Camera {
    private World worldSpace;
    private double halfWidth;
    private double halfHeight;
    private int width;
    private int height;
    private double posX;
    private double posY;
    private double posZ;
    private double fov;
    private double horizontalRotation;
    private double verticalRotation;
    private Display viewDisplay;
    private BufferedImage image;
    private double moveDistanceScale = 0.1;


    public Camera (World worldSpace, int width, int height, double posX, double posY, double posZ, double fov, double horizontalRotation,double verticalRotation) {
        this.halfWidth = (double) width/2;
        this.halfHeight = (double) height/2;
        this.width = width;
        this.height = height;
        this.posX = posX;
        this.posY = posY;
        this.posZ = posZ;
        this.fov = fov;
        this.horizontalRotation = horizontalRotation;
        this.verticalRotation = verticalRotation;
        this.worldSpace = worldSpace;
        this.worldSpace.addCamera(this);
    }

    

    public void renderCameraView() {
        this.image = new BufferedImage(this.width, this.height, BufferedImage.TYPE_INT_RGB);
        double fovScalar = Math.tan(Math.toRadians(this.fov/2));
        double cosX = Math.cos(Math.toRadians(this.horizontalRotation));
        double cosY = Math.cos(Math.toRadians(this.verticalRotation));
        double sinX = Math.sin(Math.toRadians(this.horizontalRotation));
        double sinY = Math.sin(Math.toRadians(this.verticalRotation));
        ExecutorService executorService = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors()); // Create a thread pool with the number of available processors

        // WISH TO MULTITHREAD FOR IMPROVED PERFORMANCE
        for (int pixelX = 0; pixelX < this.width; pixelX++) {
            final int lockedPixelX = pixelX; //THIS VALUE NEEDS TO BE FIXED FOR MULTITHREADING
            executorService.submit(() -> {  //SENDS FOLLOWING CODE TO BE AUTOMATICALLY MULTITHREADED
                for (int pixelY = 0; pixelY < this.height; pixelY++) {
                    Vector intersectVector = new Vector((double)(fovScalar * (double)(lockedPixelX - halfWidth) / halfWidth), (double)((double)(halfHeight - pixelY) / (halfHeight)), 1, this.posX, this.posY, this.posZ);
                    intersectVector.normalize();
                    intersectVector.rotate(cosX, cosY, sinX, sinY);
                    double distance = 999999;
                    double tempDistance;
                    int intersectObjectType = 0;
                    Ellipse intersectEllipse = null;
                    Plane intersectPlane = null;
                    int rgb = this.worldSpace.getBackgroundColorIntValue();
                    //FIND RGB VALUE(FIND INTERSECTION IF ANY)
                    for (int i = 0; i < this.worldSpace.getNumOfEllipses(); i++) {
                        tempDistance = intersectVector.intersect(this.worldSpace.getEllipse(i));
                        if(tempDistance > 0 && tempDistance < distance) {
                            distance = tempDistance;
                            intersectObjectType = 1;
                            intersectEllipse = this.worldSpace.getEllipse(i);
                        }
                    }
                    for (int j = 0; j < this.worldSpace.getNumOfPlanes(); j++) {
                        tempDistance = intersectVector.intersect(this.worldSpace.getPlane(j));
                        if(tempDistance > 0 && tempDistance < distance) {
                            distance = tempDistance;
                            intersectObjectType = 2;
                            intersectPlane = this.worldSpace.getPlane(j);
                        }
                    }
                    if(intersectObjectType == 1) {
                        rgb = intersectEllipse.getColorIntValue();
                    } else if (intersectObjectType == 2) {
                        rgb = intersectPlane.getColorIntValue();
                    }
                    this.image.setRGB(lockedPixelX, pixelY, rgb);
                }
            });
        }

        // Shut down the executor service after all tasks are completed
        executorService.shutdown();
        while (!executorService.isTerminated()) {
            // Wait until all tasks are completed
        }
    }

    public BufferedImage getImage() {
        return this.image;
    }

    public int getWidth() {
        return this.width;
    }

    public int getHeight() {
        return this.height;
    }

    public void moveForward() {
        this.posX += this.moveDistanceScale * Math.cos(Math.toRadians(90-this.horizontalRotation));
        this.posZ += this.moveDistanceScale * Math.sin(Math.toRadians(90-this.horizontalRotation));
    }

    public void moveBackward() {
        this.posX += this.moveDistanceScale * Math.cos(Math.toRadians(-90-this.horizontalRotation));
        this.posZ += this.moveDistanceScale * Math.sin(Math.toRadians(-90-this.horizontalRotation));
    }

    public void moveLeft() {
        this.posX += this.moveDistanceScale * Math.cos(Math.toRadians(180-this.horizontalRotation));
        this.posZ += this.moveDistanceScale * Math.sin(Math.toRadians(180-this.horizontalRotation));
    }

    public void moveRight() {
        this.posX += this.moveDistanceScale * Math.cos(Math.toRadians(0-this.horizontalRotation));
        this.posZ += this.moveDistanceScale * Math.sin(Math.toRadians(0-this.horizontalRotation));
    }

    public void moveUp() {
        this.posY += 0.1;
    }

    public void moveDown() {
        this.posY -= 0.1;
    }

    public void turnHorizontally(double rotationAngle){
        this.horizontalRotation += rotationAngle;
    }

    public void turnVertically(double rotationAngle){
        this.verticalRotation = Math.max(-89,Math.min(89,this.verticalRotation + rotationAngle));
    }
}