package RayTracingPackages;

import java.util.ArrayList;


public class World {
    //ITEM TYPE:1
    private ArrayList<Camera> worldCameras = new ArrayList<>();
    //ITEM TYPE:2
    private ArrayList<Sphere> worldSpheres = new ArrayList<>();
    //ITEM TYPE:3
    private ArrayList<Ellipse> worldEllipses = new ArrayList<>();
    //ITEM TYPE:4
    private ArrayList<Plane> worldPlanes = new ArrayList<>();
    private Color backgroundColor;


    public World(int backgroundColorR, int backgroundColorG, int backgroundColorB) {
        this.backgroundColor = new Color(backgroundColorR,backgroundColorG,backgroundColorB);
    }

    public World(String colorName) {
        this.backgroundColor = new Color(colorName);
    }

    //GENERAL WORLD METHODS
    public int getBackgroundColorIntValue() {
        return this.backgroundColor.getColorIntValue();
    }



    //ADD ITEMS METHODS
    public void addCamera(Camera camera) {
        this.worldCameras.add(camera);
    } 

    public void addSphere(Sphere sphere) {
        this.worldSpheres.add(sphere);
    }

    public void addEllipse(Ellipse ellipse) {
        this.worldEllipses.add(ellipse);
    }

    public void addPlane(Plane plane) {
        this.worldPlanes.add(plane);
    }

    //NUM OF ITEMS METHODS
    public int getNumOfCameras() {
        return this.worldCameras.size();
    }

    public int getNumOfSpheres() {
        return this.worldSpheres.size();
    }

    public int getNumOfEllipses() {
        return this.worldEllipses.size();
    }

    public int getNumOfPlanes() {
        return this.worldPlanes.size();
    }

    //ITEM RETRIEVAL METHODS   (NOTE:NUMBER FOR ITEMS STARTS AT 0)
    public Camera getCamera(int num) {
        return this.worldCameras.get(num);
    }

    public Sphere getSphere(int num) {
        return this.worldSpheres.get(num);
    }

    public Ellipse getEllipse(int num) {
        return this.worldEllipses.get(num);
    }

    public Plane getPlane(int num) {
        return this.worldPlanes.get(num);
    }

}