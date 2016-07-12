package com.trimblet.opencl.utilities;

import java.util.ArrayList;
import java.util.List;

public final class Utilities {

	private Utilities() {}

	public static final void printArray(float[] array) {
		List<Float> output = new ArrayList<Float>();
		for (float item: array) {
			output.add(item);
		}
		System.out.println(output);
	}

	public static final float[] newTestArray(int n) {
		float[] result = new float[n+1];
		for (int i=0; i <= n; i++) {
			result[i] = i;
		}
		return result;
	}

}
