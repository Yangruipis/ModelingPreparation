using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace 期权计算器
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private static double CumDensity(double z)
        {
            double p = 0.3275911;
            double a1 = 0.254829592;
            double a2 = -0.284496736;
            double a3 = 1.421413741;
            double a4 = -1.453152027;
            double a5 = 1.061405429;

            int sign;
            if (z < 0.0)
                sign = -1;
            else
                sign = 1;

            double x = Math.Abs(z) / Math.Sqrt(2.0);
            double t = 1.0 / (1.0 + p * x);
            double erf = 1.0 - (((((a5 * t + a4) * t) + a3)
              * t + a2) * t + a1) * t * Math.Exp(-x * x);
            return 0.5 * (1.0 + sign * erf);
        }

        private double get_value(double[] double_array )
        {
            double underlying_price = double_array[0];
            double strike_price = double_array[1];
            double due_time = double_array[2];
            double rate = double_array[3];
            double vol = double_array[4];

            double d1 = (Math.Log(underlying_price / strike_price) + (rate + Math.Pow(vol, 2) / 2) * due_time) / (vol * Math.Sqrt(due_time));
            double d2 = d1 - vol * Math.Sqrt(due_time);
            return underlying_price * CumDensity(d1) - strike_price * Math.Exp(-rate * due_time) * CumDensity(d2);
        }

        private List<int> has_one_null(string[] str)
        {
            List<int> null_index = new List<int>();
            for (int i = 0; i < str.Length; i++)
            {
                if (str[i].Trim() == string.Empty)
                {
                    null_index.Add(i);
                }
            }
            return null_index;
        }

        private double dichotomy_cal(int index, double upper, double lower, double[] param, bool positive, double price)
        {
            param[index] = (upper + lower) / 2.0;
            double pre;
            int count = 0;
            while (true) {
                count++;
                double price1 = get_value(param);
                pre = param[index];
                if (positive)
                {
                    if (price1 < price)
                    {
                        lower = param[index];
                        param[index] = (lower + upper) / 2.0;
                    }
                    else if (price1 > price)
                    {
                        upper = param[index];
                        param[index] = (lower + upper) / 2.0;
                    }
                    else
                        return param[index];
                    if (Math.Abs(param[index] - pre) < 1e-5)
                        return param[index];
                    if(count > 10000)
                        return 9999;
                }
                else
                {
                    if (price1 > price)
                    {
                        lower = param[index];
                        param[index] = (lower + upper) / 2.0;
                    }
                    else if (price1 < price)
                    {
                        upper = param[index];
                        param[index] = (lower + upper) / 2.0;
                    }
                    else
                        return param[index];
                    if (Math.Abs(param[index] - pre) < 1e-5)
                        return param[index];
                    if (count > 10000)
                        return 9999;
                }

            }
        }


        private double up, sp, t, r, vol, p;



        private void button1_Click(object sender, EventArgs e)
        {
            textBox_result.Text = "";

            string underlying_price = textBox_up.Text;
            string strike_price = textBox_sp.Text;
            string due_time = textBox_t.Text;
            string rate = textBox_r.Text;
            string volatity = textBox_v.Text;
            string price = textBox_price.Text;

            up = sp = t = r = vol = p = 9999;

            if (underlying_price != String.Empty)
                up = Convert.ToDouble(underlying_price);
            if (strike_price != String.Empty)
                sp = Convert.ToDouble(strike_price);
            if (due_time != String.Empty)
                t = Convert.ToDouble(due_time) / 365;
            if (rate != String.Empty)
                r = Convert.ToDouble(rate);
            if (volatity != String.Empty)
                vol = Convert.ToDouble(volatity);
            if (price != String.Empty)
                p = Convert.ToDouble(price);
            

            string[] string_array = new string[6] { underlying_price, strike_price, due_time, rate, volatity, price };
            var null_index = has_one_null(string_array);

            if (null_index.Count != 1)
            {
                MessageBox.Show("只能空一个待求参数！");
            }
            else
            {
                switch (null_index[0])
                {
                    case 5:
                        var result5 = get_value(new double[] { up, sp, t, r, vol });
                        textBox_result.Text = Math.Round(result5, 4).ToString();
                        break;
                    case 4:
                        // vol
                        var result4 = dichotomy_cal(4, 100, 0, new double[] { up, sp, t, r, vol }, true, p);
                        if (result4 != 9999)
                            textBox_result.Text = Math.Round(result4, 4).ToString();
                        else
                            MessageBox.Show("深度实值认购期权隐波无解！");
                        break;
                    case 3:
                        // rate
                        var result3 = dichotomy_cal(3, 1.0, 0, new double[] { up, sp, t, r, vol }, true, p);
                        textBox_result.Text = Math.Round(result3, 4).ToString();
                        break;
                    case 2:
                        // due_time
                        var result2 = dichotomy_cal(2, 100, 0, new double[] { up, sp, t, r, vol }, true, p);
                        textBox_result.Text = Math.Round(result2 * 365).ToString();
                        break;
                    case 1:
                        //strike price
                        var result1 = dichotomy_cal(1, 10, 0, new double[] { up, sp, t, r, vol }, false, p);
                        textBox_result.Text = Math.Round(result1, 4).ToString();
                        break;
                    case 0:
                        //underlying price
                        var result0 = dichotomy_cal(0, 10, 0, new double[] { up, sp, t, r, vol }, true, p);
                        textBox_result.Text = Math.Round(result0, 4).ToString();
                        break;
                }
            }



        }
    }
}
