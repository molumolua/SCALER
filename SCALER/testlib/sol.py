

import java.io.*;
import java.util.*;
import java.lang.*;
public class code2{
    static class FastScanner {
        BufferedReader br;
        StringTokenizer st;
        FastScanner(InputStream stream) {
            try {
                br = new BufferedReader(new
                    InputStreamReader(stream));
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        String n() {
            while (st == null || !st.hasMoreTokens()) {
                try {
                    st = new StringTokenizer(br.readLine());
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            return st.nextToken();
        }

        String nL() {
            String str = "";
            try
            {
                str = br.readLine();
            }
            catch (IOException e)
            {
                e.printStackTrace();
            }
            return str;
        }

        int nI() {
            return Integer.parseInt(n());
        }
        long nLo() {
            return Long.parseLong(n());
        }
     }
    private static long gcd(long a,long b) {
    	if(b==0)
    		return a;
    	else
    		return gcd(b,a%b);
    }
	public static void main(String args[]) {
		try {
			FastScanner sc=new FastScanner(System.in);
			int tests=sc.nI();
			for(int l=0;l<tests;l++) {
				long n=sc.nLo();
				long p=0;
				long sum;
				if(n%2==0)
				sum=0;
				else
					sum=1;
				
				while(n>1) {
				String s=Long.toBinaryString(n);
				int k=s.length();
				p=(long)Math.pow(2, k)-1;
				long t=(long)Math.pow(2, k-1);
				sum=sum+p;
				n=n%(t);
				}
				
				System.out.println(sum);
			}
		}
		catch(Exception e) {}
	
}
}


