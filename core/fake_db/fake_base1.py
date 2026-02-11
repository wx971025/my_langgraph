import random
from faker import Faker
from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.engine import Engine

from core.base import BaseMySQL


Base = declarative_base()

class SalesData(Base):
    __tablename__ = 'sales_data'
    sales_id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey('product_information.product_id'))
    employee_id = Column(Integer)  # 示例简化，未创建员工表
    customer_id = Column(Integer, ForeignKey('customer_information.customer_id'))
    sale_date = Column(String(50))
    quantity = Column(Integer)
    amount = Column(Float)
    discount = Column(Float)

class CustomerInformation(Base):
    __tablename__ = 'customer_information'
    customer_id = Column(Integer, primary_key=True)
    customer_name = Column(String(50))
    contact_info = Column(String(50))
    region = Column(String(50))
    customer_type = Column(String(50))

class ProductInformation(Base):
    __tablename__ = 'product_information'
    product_id = Column(Integer, primary_key=True)
    product_name = Column(String(50))
    category = Column(String(50))
    unit_price = Column(Float)
    stock_level = Column(Integer)

class CompetitorAnalysis(Base):
    __tablename__ = 'competitor_analysis'
    competitor_id = Column(Integer, primary_key=True)
    competitor_name = Column(String(50))
    region = Column(String(50))
    market_share = Column(Float)


def insert_fake_data(engine: Engine):
    Session = sessionmaker(bind=engine)
    session = Session()

    fake = Faker()

    for _ in range(50):
        customer = CustomerInformation(
            customer_name=fake.name(),
            contact_info=fake.phone_number(),
            region=fake.state(),  # 地区
            customer_type=random.choice(['Retail', 'Wholesale'])  # 零售、批发
        )
        session.add(customer)

    for _ in range(20):  # 生成20种产品
        product = ProductInformation(
            product_name=fake.word(),
            category=random.choice(['Electronics', 'Clothing', 'Furniture', 'Food', 'Toys']),  # 电子设备，衣服，家具，食品，玩具
            unit_price=random.uniform(10.0, 1000.0),
            stock_level=random.randint(10, 100)  # 库存
        )
        session.add(product)

    for _ in range(10):  # 生成10个竞争对手
        competitor = CompetitorAnalysis(
            competitor_name=fake.company(),
            region=fake.state(),
            market_share=random.uniform(0.01, 0.2)  # 市场占有率
        )
        session.add(competitor)

    session.commit()

    for _ in range(100):
        sale = SalesData(
            product_id=random.randint(1, 20),
            employee_id=random.randint(1, 10),  # 员工ID范围
            customer_id=random.randint(1, 50),
            sale_date=fake.date_between(start_date='-1y', end_date='today').strftime('%Y-%m-%d'),
            quantity=random.randint(1, 10),
            amount=random.uniform(50.0, 5000.0),
            discount=random.uniform(0.0, 0.15)
        )
        session.add(sale)

    session.commit()
    session.close()


if __name__ == "__main__":
    base_mysql = BaseMySQL()
    # base_mysql.connect_to_db("langchain_agent", auto_create=True)
    # engine = base_mysql.engine
    # Base.metadata.create_all(base_mysql.engine)
    # insert_fake_data(engine)
    base_mysql.drop_db("langchain_agent")
