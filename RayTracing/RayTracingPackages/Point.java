package RayTracingPackages;

public class Point {
    private double x;
    private double y;
    private double z;

    public Point(double x, double y, double z) {
        this.x = x;
        this.y = y;
        this.z = z;
    }

    public void setX(double newX) {
        this.x = newX;
    }

    public void setY(double newY) {
        this.y = newY;
    }

    public void setZ(double newZ) {
        this.z = newZ;
    }

    public double getX() {
        return this.x;
    } 

    public double getY() {
        return this.y;
    } 

    public double getZ() {
        return this.z;
    }

    public String toString() {
        return "( " + String.format("%3f",this.x) + ", " + String.format("%3f",this.y) + ", " + String.format("%3f",this.z) + " )";
    }
}