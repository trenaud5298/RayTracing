package RayTracingPackages;

import java.util.Random;
import java.util.ArrayList;
import java.util.List;

public class Vector {
    private double directionX;
    private double directionY;
    private double directionZ;
    private double originX;
    private double originY;
    private double originZ;
    private double length;
    private int intersectItemType;
    private int intersectItemNum;

    public Vector(double directionX, double directionY, double directionZ, double originX, double originY, double originZ) {
        this.length = Math.sqrt(directionX*directionX + directionY*directionY + directionZ*directionZ); 
        this.directionX = directionX;
        this.directionY = directionY;
        this.directionZ = directionZ;
        this.originX = originX;
        this.originY = originY;
        this.originZ = originZ;
    }


    
    //Vector Methods
    public void normalize() {
        this.directionX /= this.length;
        this.directionY /= this.length;
        this.directionZ /= this.length;
        this.length = 1;
    }

    public void rotate(double cosX, double cosY, double sinX, double sinY) {
        double tempZ = this.directionY * sinY + this.directionZ * cosY;
        this.directionY = this.directionY * cosY - this.directionZ * sinY;
        this.directionZ = tempZ * cosX - this.directionX * sinX;
        this.directionX = this.directionX * cosX + tempZ * sinX;
    }

    public void rotateX(double angleDegrees) {
        double angleRadians = Math.toRadians(angleDegrees);
        double newY = this.directionY * Math.cos(angleRadians) - this.directionZ * Math.sin(angleRadians);
        double newZ = this.directionY * Math.sin(angleRadians) + this.directionZ * Math.cos(angleRadians);
        this.directionY = newY;
        this.directionZ = newZ;
    }

    public void rotateY(double angleDegrees) {
        double angleRadians = Math.toRadians(angleDegrees);
        double newX = this.directionX * Math.cos(angleRadians) + this.directionZ * Math.sin(angleRadians);
        double newZ = -this.directionX * Math.sin(angleRadians) + this.directionZ * Math.cos(angleRadians);
        this.directionX = newX;
        this.directionZ = newZ;
    }

    public void rotateZ(double angleDegrees) {
        double angleRadians = Math.toRadians(angleDegrees);
        double newX = this.directionX * Math.cos(angleRadians) - this.directionY * Math.sin(angleRadians);
        double newY = this.directionX * Math.sin(angleRadians) + this.directionY * Math.cos(angleRadians);
        this.directionX = newX;
        this.directionY = newY;
    }

    public double dotProdcut(Vector vec2){
        return (this.directionX * vec2.directionX + this.directionY * vec2.directionY + this.directionZ * vec2.directionX);
    }

    //Vector Intersect Methods

    
    public double intersect(Ellipse ellipse) {
        //Ellipse Variables Needed for Equation
        double aSquared = ellipse.getWidth();
        aSquared *= aSquared;
        double bSquared = ellipse.getHeight();
        bSquared *= bSquared;
        double cSquared = ellipse.getLength();
        cSquared *= cSquared;
        double radius = ellipse.getRadius();
        double deltaX = this.originX - ellipse.getX();
        double deltaY = this.originY - ellipse.getY();
        double deltaZ = this.originZ - ellipse.getZ();



        double a =  ( ((this.directionX * this.directionX)/aSquared) + ((this.directionY * this.directionY)/bSquared) + ((this.directionZ * this.directionZ)/cSquared) ) ;
        double b = 2*(  ((this.directionX * deltaX )/aSquared) + ((this.directionY * deltaY )/bSquared) + ((this.directionZ * deltaZ )/cSquared)  );
        double c = ( (deltaX*deltaX/aSquared) + (deltaY*deltaY/bSquared) + (deltaZ*deltaZ/cSquared) - (radius*radius) );

        double squareRoot = b*b - 4*a*c;
        if (squareRoot < 0) {
            return 0;
        } else if (squareRoot == 0) {
            double t1 = (-1*b + Math.sqrt(squareRoot)) / (2*a);
            return Math.max(0,t1);
            }

        double t1 = (-1*b + Math.sqrt(squareRoot)) / (2*a);
        double t2 = (-1*b - Math.sqrt(squareRoot)) / (2*a);
        return Math.max(Math.max(0,t1),Math.max(t2,0));

        }

    
    public double intersect(Plane plane) {
        double a = plane.getA();
        double b = plane.getB();
        double c = plane.getC();
        double numerator = plane.getD() + a*(plane.getX1()-this.originX) + b*(plane.getY1()-this.originY) + c*(plane.getZ1()-this.originZ);
        double denominator = a*this.directionX + b*this.directionY + c*this.directionZ;
        double distance = (double) numerator / denominator;
        if(distance <= 0) {
            return 0.0;
        }
        double intersectPointX = this.originX+this.directionX*distance;
        if(!(plane.getLowerXBounds() < intersectPointX && intersectPointX< plane.getUpperXBounds())) {
            return 0.0;
        }
        double intersectPointY = this.originY+this.directionY*distance;
        if(!(plane.getLowerYBounds() < intersectPointY && intersectPointY < plane.getUpperYBounds())) {
            return 0.0;
        }
        double intersectPointZ = this.originZ+this.directionZ*distance;
        if(!(plane.getLowerZBounds() < intersectPointZ && intersectPointZ< plane.getUpperZBounds())) {
            return 0.0;
        }
        return distance;
        
    }

    public Vector getBounceVector(Plane plane, Point intersectionPoint) {
        //NEEDS MORE WORK
        Vector bounceVector = new Vector(1,1,1,intersectionPoint.getX(),intersectionPoint.getY(),intersectionPoint.getZ());
        return bounceVector;

    }





    //Variable Retrieving
    public double getDirectionX() {
        return this.directionX;
    }

    public double getDirectionY() {
        return this.directionY;
    }

    public double getDirectionZ() {
        return this.directionZ;
    }

    public double getOriginX() {
        return this.originX;
    }

    public double getOriginY(){
        return this.originY;
    }

    public double getOriginZ(){
        return this.originZ;
    }

    public double getLength() {
        return this.length;
    }

    public int getIntersectItemType() {
        return this.intersectItemType;
    }

    public int getIntersectItemNum() {
        return this.intersectItemNum;
    }

    //Variable Changing
    public void setDirectionX(double newDirectionX) {
        this.directionX = newDirectionX;
    }

    public void setDirectionY(double newDirectionY) {
        this.directionY = newDirectionY;
    }

    public void setDirectionZ(double newDirectionZ) {
        this.directionZ = newDirectionZ;
    }

    public void setOriginX(double newOriginX) {
        this.originX = newOriginX;
    }

    public void setOriginY(double newOriginY) {
        this.originY = newOriginY;
    }

    public void setOriginZ(double newOriginZ) {
        this.originZ = newOriginZ;
    }


    public String toString() {
        return "< " + String.format("%.3f",this.directionX) + " , " + String.format("%.3f",this.directionY) + " , " + String.format("%.3f",this.directionZ) + " >";
    }
}


