# Pro Analyzer Advanced

نسخة تشغيلية أولى لبرنامج تحليل أداء التزلج باستخدام Streamlit وMediaPipe.

## التشغيل
```bash
pip install -r requirements.txt
streamlit run app.py
```

## الميزات الحالية
- رفع فيديو وتحليل أولي للحركة
- قياس زاوية الركبة والجذع
- تصنيف أخطاء إلى: critical / major / minor / good
- حساب أولي لـ Base Value و GOE و Technique Score
- تنزيل تقارير JSON و Excel و TXT

## الخطوات التالية
- إضافة كشف مراحل الحركة
- تحسين نموذج النقاط ليتماشى أكثر مع التحكيم الرسمي
- دعم التحليل المباشر من الكاميرا
- إضافة قاعدة بيانات عناصر أوسع
